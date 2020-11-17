import os
import sys
import torch
import random
import torch.nn.functional as F
from torch.distributions.normal import Normal

import layers
from transformer_xl import StableTransformerXL

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset_env'))
import file_storage

def cosine_search(config, deg, word_embd):
    cos_sim = torch.nn.CosineSimilarity()
    all_instructions = file_storage.get_instruct_traj() # NOTE: Heavy, try alternative. 
    max_index, max_sim = -1, config.max_cosine_cutoff
    for j, (search_instruction, trajectory) in enumerate(all_instructions):
        search_words = search_instruction['word_embeddings']
        for k in range(search_words.shape[0]):
            word2 = search_words[k]
            word, word2 = torch.reshape(word_embd, (1, -1)), torch.reshape(word2, (1, -1))
            sim = cos_sim(word, word2)

            if sim > max_sim:
                max_sim = sim
                max_index = j
                break
        
        if max_index != -1:
            break

    traj = all_instructions[max_index][1]
    return traj

def get_similar_traj(config, deg, instruct_traj):
    instruction = instruct_traj['instruction']
    traj_batch = []
    for i in range(len(instruction)):
        traj_seq = []
        for j, word in enumerate(instruction[i].split(" ")):
            search_space = deg.config.instruct_db.objects.filter(instruction__contains=word.lower())
            if search_space:
                episode_id = random.choice(search_space).trajectory.episode_id
                traj = file_storage.get_trajectory_non_db(episode_id=str(episode_id)) if config.no_db else file_storage.get_trajectory(episode_id=episode_id)
            else:
                word_embd = instruct_traj[deg.word_embeddings_key][i][j]
                traj = cosine_search(config, word_embd)
            traj_seq.append(traj)
        traj_seq = deg._collate_wrap(remove_task_state=True)(traj_seq)
        traj_seq = {key : torch.reshape(traj_seq[key], (traj_seq[key].shape[0] * traj_seq[key].shape[1], -1)) for key in traj_seq.keys()}
        traj_seq = {key : traj_seq[key].cpu().detach().numpy() for key in traj_seq.keys()}
        traj_batch.append(traj_seq)
    traj_batch = deg._collate_wrap(remove_task_state=True)(traj_batch)
    return traj_batch

class PerceptionModule(torch.nn.Module):
    '''
        Maps raw observation (image & proprioception) O_t to a low dimension embedding s_t. 
        TODO : Normalize proprioception to have zero mean and unit variance.
    '''
    def __init__(self, visual_obv_dim=[256, 256, 3], dof_obv_dim=[8], state_dim=64):
        super(PerceptionModule, self).__init__()
        
        self.visual_obv_dim = visual_obv_dim
        self.dof_obv_dim = dof_obv_dim
        self.state_dim = state_dim

        assert visual_obv_dim[0] == visual_obv_dim[1]
        # NOTE/TODO : IMPORTANT - whenever the dim. of the obvs changes, this has to be changed
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(8, 8), stride=4, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)
        self.ss = layers.SpatialSoftmax(height=13, width=13, channel=64) # NOTE : Change dim here
        self.lin1 = torch.nn.Linear(128, 512) # SS O/Ps shape (N, C * 2)
        self.lin2 = torch.nn.Linear(512, state_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, visual_obv, dof_obv=None):
        output = self.relu(self.conv1(visual_obv))
        output = self.relu(self.conv2(output))
        output = self.conv3(output)
        output = self.ss(output)
        output = self.relu(self.lin1(output))
        output = self.lin2(output)

        if dof_obv is not None:
            output = torch.cat((output, dof_obv), -1)
        return output

class BasicAttnModel(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BasicAttnModel, self).__init__()

        self.attn = layers.RLMultiHeadAttention(state_dim, action_dim, nhead=1)
        self.projection = torch.nn.Linear(state_dim, action_dim)

    def forward(self, curr_state, state_set, action_set, mask=None):
        # TODO - IMPORTANT : See if attention is being applied over batch - wrong if only over seqs.
        # If seqs, reshape to have batch_size=1
        output = self.attn(curr_state, state_set, action_set)
        output = self.projection(output)
        return output

class RLTransformerEncoder(torch.nn.Module):
    '''
        Passes the state-vectors through number of self-attention layers.
        Then applies RLTransofmerEncoderLayer over the state and action vectors.
        TODO : Make it stochastic
    '''
    def __init__(self, state_dim, action_dim, n_state_encoders=4, n_heads=1):
        super(RLTransformerEncoder, self).__init__()
        self.state_self_attn = StableTransformerXL(d_input=state_dim, n_layers=n_state_encoders, n_heads=n_heads, d_head_inner=32, d_ff_inner=64)
        self.final_attn = layers.RLTransorfmerEncoderLayer(state_dim, action_dim, n_heads=n_heads)
        self.action_projection = torch.nn.Linear(state_dim, action_dim)

        self.state_set_memory = None
        self.curr_state_memory = None

    def forward(self, curr_state, state_set, action_set, mask=None):
        state_set = self.state_self_attn(state_set, self.state_set_memory)
        curr_state = self.state_self_attn(curr_state, self.curr_state_memory) #TODO , mask=mask)

        state_set, self.state_set_memory = state_set['logits'], state_set['memory']
        curr_state, self.curr_state_memory = curr_state['logits'], curr_state['memory']

        pred_action = self.final_attn(curr_state, state_set, action_set, mask=None)
        pred_action = self.action_projection(pred_action)
        return pred_action

class TransformerGaussianPolicy(torch.nn.Module):
    def __init__(self, state_dim, act_dim, n_transformer_layers=4, n_attn_heads=3):
        ''' 
            NOTE - I/P Shape : [seq_len, batch_size, state_dim]
        '''
        super(TransformerGaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.transformer = StableTransformerXL(d_input=state_dim, n_layers=n_transformer_layers, 
            n_heads=n_attn_heads, d_head_inner=32, d_ff_inner=64)
        self.memory = None

        self.head_sate_value = torch.nn.Linear(state_dim, 1)
        self.head_act_mean = torch.nn.Linear(state_dim, act_dim)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

    def _distribution(self, trans_state):
        mean = self.tanh(self.head_act_mean(trans_state))
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def _log_prob_from_distribution(self, policy, action):
        return policy.log_prob(action).sum(axis=-1) 

    def forward(self, state, action=None):
        trans_state = self.transformer(state, self.memory)
        trans_state, self.memory = trans_state['logits'], trans_state['memory']

        policy = self._distribution(trans_state)
        state_value = self.head_sate_value(trans_state)

        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(policy, action)

        return policy, logp_a, state_value

    def step(self, state):
        if state.shape[0] == self.state_dim:
            state = state.reshape(1, 1, -1)
        with torch.no_grad():
            trans_state = self.transformer(state, self.memory)
            trans_state, self.memory = trans_state['logits'], trans_state['memory']

            policy = self._distribution(trans_state)
            action = policy.sample()
            logp_a = self._log_prob_from_distribution(policy, action)
            state_value = self.head_sate_value(trans_state)

        return action.numpy(), logp_a.numpy(), state_value.numpy()
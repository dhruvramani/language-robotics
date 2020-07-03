import numpy as np
import torch
import torch.nn.Functional as F
from layers import SpatialSoftmax, ConditionalVAE, SeqVAE

# NOTE > `self.perception_module` and other `torch.nn.Module`s passed as args
#       All variables assigned point to the same memory location in Python.
#       So when the original object is modified, even the assigned value get's modified (eg. by the grad).

class PerceptionModule(torch.nn.Module):
    '''
        Maps raw observation (image & proprioception) O_t to a low dimension embedding s_t. 
        TODO : Normalize proprioception to have zero mean and unit variance.
    '''
    def __init__(self, visual_obv_dim=[200, 200, 3], dof_obv_dim=[8], state_dim=64):
        super(self, PerceptionModule).__init__()
        
        self.visual_obv_dim = visual_obv_dim
        self.dof_obv_dim = dof_obv_dim
        self.state_dim = state_dim

        assert self.visual_obv_dim[0] == self.visual_obv_dim[1]

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(8, 8), stride=4, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1)
        self.ss = SpatialSoftmax(22, 22, 64)
        self.lin1 = torch.nn.Linear(22 * 22* 64, 512)
        self.lin2 = torch.nn.Linear(512, self.state_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, visual_obv, dof_obv=None):
        output = self.relu(self.conv1(visual_obv))
        output = self.relu(self.conv2(output))
        output = self.conv3(output)
        output = self.ss(output)
        output = torch.flatten(output)
        output = self.relu(self.lin1(output))
        output = self.lin2(output)

        if dof_obv:
            dof_obv = torch.flatten(dof_obv)
            output = torch.cat((output, dof_obv), -1)

        return output

class VisualGoalEncoder(torch.nn.Module):
    '''
        Maps goal s_g to latent embedding z.
        + NOTE First pass through the PerceptionModule to get s_g = P(O_{-1}).
        + NOTE Pass only the visual observation.
    '''
    def __init__(self, perception_module, state_dim=64, goal_dim=32):
        super(self, VisualGoalEncoder).__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.lin1 = torch.nn.Linear(state_dim, 2048)
        self.relu = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(2048, 2048)
        self.lin3 = torch.nn.Linear(2048, goal_dim)
        # TODO : Might have to model it as a gaussian distribution. 

        self.perception_module = perception_module

    def forward(self, visual_obv):
        if visual_obv.size() == 2 # To filter dof-obvs
            visual_obv = visual_obv[0]
        if(visual_obv.size([-1]) != self.state_dim):
            visual_obv = self.perception_module(visual_obv, None)
        output = self.relu(self.lin1(visual_obv))
        output = self.relu(self.lin2(output))
        output = self.lin3(output)

        return output

class PlanRecognizerModule(torch.nn.Module):
    ''' 
        A SeqVAE which maps the entire trajectory T to latent space z_p ~ q(z_p|T).
        Represents the posterior distribution over z_p.
        NOTE : Pass a *single* trajectory [[[Ot/st, at], ... K]] only, where Ot = [Vobv, DOF].
        NOTE : Trajectory shape : (1, K, 2) (Batch Size = 1).
    '''
    def __init__(self, sequence_length, perception_module, state_dim=72, latent_dim=256):
        super(self, PlanRecognizerModule).__init__()
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.perception_module = perception_module
        self.seqVae = SeqVAE(self.sequence_length, self.state_dim, latent_size=self.latent_dim)

    def forward(self, trajectory):
        assert trajectory.size()[0] == 1 and trajectory.size()[1] == self.sequence_length
        if trajectory[0][0][0].size()[-1] != self.state_dim:
            for i in self.sequence_length:
                visual_obv, dof_obv = trajectory[0][i][0]
                trajectory[0][i][0] = self.perception_module(visual_obv, dof_obv)

        for i in self.sequence_length:
            trajectory[0][i] = torch.cat((trajectory[0][i][0], trajectory[0][i][1]), -1)

        z, mean, logv = self.seqVae(trajectory)
        return z, mean, logv

class PlanProposerModule(torch.nn.Module):
    ''' 
        A ConditionalVAE which maps initial state s_0 and goal z to latent space z_p ~ p(z_p|s_0, z).
        Represents a prior over z_p.
    '''
    def __init__(self, goal_encoder, perception_module, state_dim=72, goal_dim=32, latent_dim=256):
        super(self, PlanProposerModule).__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.latent_dim = latent_dim

        self.perception_module = perception_module
        self.cVae = ConditionalVAE(state_size + goal_dim, latent_dim)

    def forward(self, initial_obv, goal_obv, goal_encoder):
        if initial_obv.size()[-1] != self.state_dim:
            visual_obv, dof_obv = initial_obv
            initial_obv = self.perception_module(visual_obv, dof_obv)

        if goal_obv.size()[-1] != self.goal_dim:
            if(type(goal_encoder) == VisualGoalEncoder):
                goal_obv = goal_encoder(goal_obv, self.perception_module)

        z, mean, logv = self.cVae(initial_obv, goal_obv)
        return z, mean, logv

class ControlModule(torch.nn.Module):
    ''' 
        RNN based goal (z), z_p conditioned policy : a_t ~ \pi(a_t | s_t, z, z_p).
    '''
    def __init__(self, perception_module, action_dim=8, state_dim=72, goal_dim=32, latent_dim=256,
                 hidden_size=2048, batch_size=1, rnn_type='RNN', num_layers=2,
                 mix_size=10, ):
        super(self, ControlModule).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.input_dim = self.state_dim + self.goal_dim + self.latent_dim
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.upper()
        self.num_layers = num_layers

        assert self.rnn_type in ['LSTM', 'GRU', 'RNN']
        self.rnn = {'LSTM' : torch.nn.LSTMCell, 'GRU' : torch.nn.GRUCell, 'RNN' : torch.nn.RNNCell}[self.rnn_type]

        self.perception_module = perception_module

        self.rnn1 = self.rnn(self.input_dim, self.hidden_size)
        self.rnn2 = self.rnn(self.hidden_size, self.hidden_size)

        # NOTE : Original paper used a Mixture of Logistic (MoL) dist. Implement later.
        self.hidden2mean = torch.Linear(self.hidden_size, self.action_dim)
        log_std = -0.5 * np.ones(self.action_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # The hidden states of the RNN.
        self.h1 = torch.randn(self.batch_size, self.hidden_size)
        self.h2 = torch.randn(self.batch_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tah()

    def _prepare_obs(self, state, goal, zp, goal_encoder):
        if state.size()[-1] != self.state_dim:
            state = self.perception_module(state)
        if goal.size()[-1] != self.goal_dim:
            if(type(goal_encoder) == VisualGoalEncoder):
                goal = goal_encoder(goal, self.perception_module)

        obs = torch.cat((state, goal, zp), -1)
        return obs

    def _distribution(self, obs):
        self.h1 = self.relu(self.rnn1(obs, self.h1)) 
        self.h2 = self.relu(self.rnn2(self.h1, self.h2))

        mean = self.tanh(self.hidden2mean(self.h2))
        std = torch.exp(self.log_std)
        return torch.distributions.normal(mean, std)

    def _log_prob_from_distribution(self, policy, action):
        return policy.log_prob(action).sum(axis=-1) 

    def forward(self, state, goal, zp, goal_encoder, action=None):
        obs = self._prepare_obs(state, goal, zp, goal_encoder)
        policy = self._distribution(obs)
        
        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(policy, action)

        return policy, logp_a

    def step(self, state, goal, zp, goal_encoder):
        with torch.no_grad():
            obs = self._prepare_obs(state, goal, zp, goal_encoder)
            policy = self._distribution(obs)
            action = policy.sample()
            logp_a = self._log_prob_from_distribution(policy, action)

        return action.numpy(), logp_a.numpy()

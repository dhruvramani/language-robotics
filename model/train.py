import os
import torch
import numpy as np
import torch.nn.Functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter

import utils
from models import *

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def train_visual_goals(env_fn, dataset, config):
    '''
        > Train on batches of random play sequences : 
            For each training batch and each batch sequence element: 
            Sample a K-length sequence of observations and actions T. 
            Extract the final observation in T as the synthetic goal state O_g and encode it : s_g.  
    '''

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env, obs_dims, act_dim = env_fn()

    perception_module = PerceptionModule(obs_dims[0], obs_dims[1]).to(device)
    visual_goal_encoder = VisualGoalEncoder().to(device)
    plan_recognizer = PlanRecognizerModule(config.max_sequence_length).to(device)
    plan_proposer = PlanProposerModule()
    control_module = ControlModule()

    params = list(perception_module.parameters()) + list(visual_goal_encoder.parameters())
    params += list(plan_recognizer.parameters()) + list(plan_proposer.parameters())
    params += list(control_module.parameters())
    print("Number of parameters : {}".format(len(params)))

    kl_loss = torch.nn.KLDivLoss()
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    # TODO : Change code to make it work with bigger batch-sizes.
    # TODO : Put the random trajectory size setting in dataset.py    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    tensorboard_writer = SummaryWriter()

    for epoch in tqdm(range(config.max_epochs), desc="Epochs"):
        for i, trajectory in enumerate(data_loader):
            trajectory = torch.FloatTensor(trajectory).to(device)

            assert trajectory.size()[0] == 1 and trajectory.size()[-1] == 2 
            # Batch size = 1, last-dim = 2 as (state, action)

            last_obvs, last_action = trajectory[0, -1, :]
            assert last_obvs.size()[0] == config.num_obv_types
            goal_state = perception_module(last_obvs[0])
            goal_state = visual_goal_encoder(goal_state)

            visual_obvs, dof_obs = trajectory[0, :, 0]
            trajectory[0, :, 0] = perception_module(visual_obvs, dof_obs)

            prior_z, prior_mean, prior_logv = plan_proposer(trajectory[0, 0, 0], goal_state)
            post_z, post_mean, post_logv = plan_recognizer(trajectory)

            goal_states = goal_state.repeat(trajectory.size()[1], goal_state.size()[1:])
            post_zs = post_z.repeat(trajectory.size()[1], post_z.size()[1:])
            pi, logp_a = control_module(state=trajectory[0, :, 0], goal=goal_states, 
                            zp=post_zs, action=trajectory[0, :, 1])

            optimizer.zero_grad()

            loss_pi = -logp_a.mean(dim=-1)
            loss_zp = kl_loss(post_z, prior_z) 
            loss = loss_pi + config.beta * loss_zp
            loss = loss.mean()

            loss.backward()
            optimizer.step()





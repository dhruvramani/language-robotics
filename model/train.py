import os
import torch
import numpy as np
import torch.nn.Functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter

import utils
from models import *

# NOTE : If in future, you operate on bigger hardwares - move to PyTorch Lightning
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def train_visual_goals(deg, config):
    '''
        > Train on batches of random play sequences : 
            For each training batch and each batch sequence element: 
            Sample a K-length sequence of observations and actions T. 
            Extract the final observation in T as the synthetic goal state O_g and encode it : s_g.  
    '''

    tensorboard_writer = SummaryWriter(logdir=config.tensorboard_path)
    env = deg.get_env()
    obs_dims, act_dim = deg.obs_space['camera'], deg.action_space # TODO : DEBUG here

    perception_module = PerceptionModule(obs_dims[0], obs_dims[1], config.visual_state_dim).to(device)
    visual_goal_encoder = VisualGoalEncoder(config.visual_state_dim, config.goal_dim).to(device)
    plan_recognizer = PlanRecognizerModule(config.max_sequence_length, config.combined_state_dim, config.latent_dim).to(device)
    plan_proposer = PlanProposerModule(config.combined_state_dim, config.goal_dim, config.latent_dim).to(device)
    control_module = ControlModule(act_dim, config.combined_state_dim, config.goal_dim, config.latent_dim).to(device)

    params = list(perception_module.parameters()) + list(visual_goal_encoder.parameters())
    params += list(plan_recognizer.parameters()) + list(plan_proposer.parameters())
    params += list(control_module.parameters())
    print("Number of parameters : {}".format(len(params)))

    if(config.save_graphs):
        tensorboard_writer.add_graph(perception_module)
        tensorboard_writer.add_graph(visual_goal_encoder)
        tensorboard_writer.add_graph(plan_recognizer)
        tensorboard_writer.add_graph(plan_proposer)
        tensorboard_writer.add_graph(control_module)

    kl_loss = torch.nn.KLDivLoss()
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    if(config.resume):
        perception_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'perception.pth')))
        visual_goal_encoder.load_state_dict(torch.load(os.path.join(config.models_save_path, 'visual_goal.pth')))
        plan_recognizer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'plan_recognizer.pth')))
        plan_proposer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'plan_proposer.pth')))
        control_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'control_module.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'optimizer.pth')))

    # TODO *IMPORTANT* : Change code to make it work with bigger batch-sizes.  
    data_loader = DataLoader(deg.dataset, batch_size=config.batch_size, shuffle=True, num_workers=1)

    for epoch in tqdm(range(config.max_epochs), desc="Check Tensorboard"):
        for i, trajectory in enumerate(data_loader):
            trajectory = torch.FloatTensor(trajectory).to(device)

            # Batch size = 1, last-dim = 2 as (state, action)
            assert trajectory.size()[0] == 1 and trajectory.size()[-1] == 2 
            
            last_obvs = trajectory[0, -1, 0]
            assert last_obvs.size()[0] == config.num_obv_types
            goal_state = perception_module(last_obvs[0])
            goal_state = visual_goal_encoder(goal_state)

            visual_obvs, dof_obs = trajectory[0, :, 0]
            trajectory[0, :, 0] = perception_module(visual_obvs, dof_obs)
            inital_state = trajectory[0, 0, 0]

            prior_z, prior_mean, prior_logv = plan_proposer(inital_state, goal_state)
            post_z, post_mean, post_logv = plan_recognizer(trajectory)

            goal_states = goal_state.repeat(trajectory.size()[1], goal_state.size()[-1])
            post_zs = post_z.repeat(trajectory.size()[1], post_z.size()[-1])
            pi, logp_a = control_module(state=trajectory[0, :, 0], goal=goal_states, 
                            zp=post_zs, action=trajectory[0, :, 1])

            optimizer.zero_grad()

            loss_pi = -logp_a.mean(dim=-1)
            loss_zp = kl_loss(post_z, prior_z) 
            loss = loss_pi + config.beta * loss_zp
            loss = loss.mean()

            tensorboard_writer.add_scalar('Policy Loss', loss_pi)
            tensorboard_writer.add_scalar('KL Loss', loss_zp)
            tensorboard_writer.add_scalar('Total Loss', loss)

            loss.backward()
            optimizer.step()

            if int(i % config.save_interval) == 0:
                torch.save(perception_module.state_dict(), os.path.join(config.models_save_path, 'perception.pth'))
                torch.save(visual_goal_encoder.state_dict(), os.path.join(config.models_save_path, 'visual_goal.pth'))
                torch.save(plan_recognizer.state_dict(), os.path.join(config.models_save_path, 'plan_recognizer.pth'))
                torch.save(plan_proposer.state_dict(), os.path.join(config.models_save_path, 'plan_proposer.pth'))
                torch.save(control_module.state_dict(), os.path.join(config.models_save_path, 'control_module.pth'))
                torch.save(optimizer.state_dict(), os.path.join(config.models_save_path, 'optimizer.pth'))


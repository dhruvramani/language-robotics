import os
import torch
import numpy as np
import torch.nn.Functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter

from models import *
from language_models import *

# NOTE : If in future, you operate on bigger hardwares - move to PyTorch Lightning
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def train_multi_context_goals(config):
    deg = config.deg()
    tensorboard_writer = SummaryWriter(logdir=config.tensorboard_path)
    env = deg.get_env()
    
    vobs_dim, dof_dim = deg.obs_space[deg.vis_obv_key], deg.obs_space[deg.dof_obv_key] 
    act_dim = deg.action_space # TODO : DEBUG here

    perception_module = PerceptionModule(vobs_dim, dof_dim, config.visual_state_dim).to(device)
    visual_goal_encoder = VisualGoalEncoder(config.visual_state_dim, config.goal_dim).to(device)
    plan_recognizer = PlanRecognizerModule(config.max_sequence_length, config.combined_state_dim, config.latent_dim).to(device)
    plan_proposer = PlanProposerModule(config.combined_state_dim, config.goal_dim, config.latent_dim).to(device)
    control_module = ControlModule(act_dim, config.combined_state_dim, config.goal_dim, config.latent_dim).to(device)

    if config.use_lang and config.use_lang_model:
        instruction_encoder = LanguageModelInstructionEncoder(config.lang_model, config.latent_dim).to(device)
    elif config.use_lang:
        instruction_encoder = BasicInstructionEncoder(config.latent_dim).to(device)

    params = list(perception_module.parameters()) + list(visual_goal_encoder.parameters())
    params += list(plan_recognizer.parameters()) + list(plan_proposer.parameters()) + list(control_module.parameters()) 
    
    if config.use_lang:
        parameters += list(instruction_encoder.goal_dist.parameters()) 

    print("Number of parameters : {}".format(len(params)))

    if(config.save_graphs):
        tensorboard_writer.add_graph(perception_module)
        tensorboard_writer.add_graph(visual_goal_encoder)
        tensorboard_writer.add_graph(plan_recognizer)
        tensorboard_writer.add_graph(plan_proposer)
        tensorboard_writer.add_graph(control_module)

        if config.use_lang:
            tensorboard_writer.add_graph(instruction_encoder)

    kl_loss = torch.nn.KLDivLoss()
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    if(config.resume):
        # TODO : IMPORTANT - Check if file exist before loading
        # TODO : Implement load & save functions within the class for easier loading and saving
        perception_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'perception.pth')))
        visual_goal_encoder.load_state_dict(torch.load(os.path.join(config.models_save_path, 'visual_goal.pth')))
        plan_recognizer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'plan_recognizer.pth')))
        plan_proposer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'plan_proposer.pth')))
        control_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'control_module.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'optimizer.pth')))

        if config.use_lang and config.use_lang_model:
            instruction_encoder.goal_dist.load_state_dict(torch.load(os.path.join(config.models_save_path, 'lang_model_{}.pth'.format(config.lang_model))))
        elif config.use_lang:
            instruction_encoder.load_state_dict(torch.load(os.path.join(config.models_save_path, 'basic_instruct_model.pth')))

    print("Tensorboard path : {}".format(config.tensorboard_path))
    visual_data_loader = DataLoader(deg.traj_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    
    if config.use_lang:
        instruct_data_loader = DataLoader(deg.instruct_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    def inference(trajectory, goal_state):
        visual_obvs, dof_obs = trajectory[0, :, 0]
        trajectory[0, :, 0] = perception_module(visual_obvs, dof_obs) # DEBUG : Might raise in-place errors
        inital_state = trajectory[0, 0, 0]

        prior_z, prior_mean, prior_logv = plan_proposer(inital_state, goal_state)
        post_z, post_mean, post_logv = plan_recognizer(trajectory)

        goal_states = goal_state.repeat(trajectory.size()[1], goal_state.size()[-1])
        post_zs = post_z.repeat(trajectory.size()[1], post_z.size()[-1])
        pi, logp_a = control_module(state=trajectory[0, :, 0], goal=goal_states, 
                        zp=post_zs, action=trajectory[0, :, 1])

        loss_pi = -logp_a.mean(dim=-1)
        loss_zp = kl_loss(post_z, prior_z) 
        loss = loss_pi + config.beta * loss_zp
        loss = loss.mean()

        return loss

    for epoch in tqdm(range(config.max_epochs * config.context_steps_scale), desc="Check Tensorboard"):
        loss_visual, loss_lang = 0.0, 0.0
        optimizer.zero_grad()

        for i in range(config.max_steps_per_context):
            trajectory = next(visual_data_loader)
            # Batch size = 1, last-dim = 2 as (state, action)
            assert trajectory.size()[0] == 1 and trajectory.size()[-1] == 2 
            
            last_obvs = trajectory[0, -1, 0]
            assert last_obvs.size()[0] == config.num_obv_types
            goal_state = perception_module(last_obvs[0])
            goal_state, _, _ = visual_goal_encoder(goal_state)

            loss = inference(trajectory, goal_state)
            loss_visual += loss

        if use_lang:
            for i in range(config.max_steps_per_context):
                instruction, trajectory = next(instruct_data_loader)
                # Batch size = 1, last-dim = 2 as (state, action)
                assert trajectory.size()[0] == 1 and trajectory.size()[-1] == 2 

                goal_state, _, _ = instruction_encoder(instruction)
                loss = inference(trajectory, goal_state)
                loss_lang += loss

        loss_visual *= (1.0 / config.max_steps_per_context)
        loss_lang *= (1.0 / config.max_steps_per_context)
        loss = 0.5 * (loss_visual + loss_lang)
        
        loss.backward()
        optimizer.step()

        tensorboard_writer.add_scalar('Visual Loss', loss_visual)
        tensorboard_writer.add_scalar('Langauge Loss', loss_lang)
        tensorboard_writer.add_scalar('Total Loss', loss)

        if int(i % config.save_interval_epoch) == 0:
            torch.save(perception_module.state_dict(), os.path.join(config.models_save_path, 'perception.pth'))
            torch.save(visual_goal_encoder.state_dict(), os.path.join(config.models_save_path, 'visual_goal.pth'))
            torch.save(plan_recognizer.state_dict(), os.path.join(config.models_save_path, 'plan_recognizer.pth'))
            torch.save(plan_proposer.state_dict(), os.path.join(config.models_save_path, 'plan_proposer.pth'))
            torch.save(control_module.state_dict(), os.path.join(config.models_save_path, 'control_module.pth'))
            torch.save(optimizer.state_dict(), os.path.join(config.models_save_path, 'optimizer.pth'))

            if config.use_lang and config.use_lang_model:
                torch.save(instruction_encoder.goal_dist.state_dict(), os.path.join(config.models_save_path, 'lang_model_{}.pth'.format(config.lang_model)))
            elif config.use_lang:
                torch.save(instruction_encoder.state_dict(), os.path.join(config.models_save_path, 'basic_instruct_model.pth'))



def train_visual_goals(config):
    '''
        This method is bascially same as train_multi_context_goals when use_lang is turned off. 
        TODO : Might delete later.
    '''
    deg = config.deg()
    tensorboard_writer = SummaryWriter(logdir=config.tensorboard_path)
    env = deg.get_env()
    
    vobs_dim, dof_dim = deg.obs_space[deg.vis_obv_key], deg.obs_space[deg.dof_obv_key] 
    act_dim = deg.action_space # TODO : DEBUG here

    perception_module = PerceptionModule(vobs_dim, dof_dim, config.visual_state_dim).to(device)
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
        # TODO : IMPORTANT - Check if file exist before loading
        perception_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'perception.pth')))
        visual_goal_encoder.load_state_dict(torch.load(os.path.join(config.models_save_path, 'visual_goal.pth')))
        plan_recognizer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'plan_recognizer.pth')))
        plan_proposer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'plan_proposer.pth')))
        control_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'control_module.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'optimizer.pth')))

    print("Tensorboard path : {}".format(config.tensorboard_path))
    # TODO *IMPORTANT* : Change code to make it work with bigger batch-sizes.  
    data_loader = DataLoader(deg.traj_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    for epoch in tqdm(range(config.max_epochs), desc="Check Tensorboard"):
        for i, trajectory in enumerate(data_loader):
            trajectory = torch.FloatTensor(trajectory).to(device)

            # Batch size = 1, last-dim = 2 as (state, action)
            assert trajectory.size()[0] == 1 and trajectory.size()[-1] == 2 
            
            last_obvs = trajectory[0, -1, 0]
            assert last_obvs.size()[0] == config.num_obv_types
            goal_state = perception_module(last_obvs[0])
            goal_state, _, _ = visual_goal_encoder(goal_state)

            visual_obvs, dof_obs = trajectory[0, :, 0]
            trajectory[0, :, 0] = perception_module(visual_obvs, dof_obs) # DEBUG : Might raise in-place errors
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

            if int(i % config.save_interval_steps) == 0:
                torch.save(perception_module.state_dict(), os.path.join(config.models_save_path, 'perception.pth'))
                torch.save(visual_goal_encoder.state_dict(), os.path.join(config.models_save_path, 'visual_goal.pth'))
                torch.save(plan_recognizer.state_dict(), os.path.join(config.models_save_path, 'plan_recognizer.pth'))
                torch.save(plan_proposer.state_dict(), os.path.join(config.models_save_path, 'plan_proposer.pth'))
                torch.save(control_module.state_dict(), os.path.join(config.models_save_path, 'control_module.pth'))
                torch.save(optimizer.state_dict(), os.path.join(config.models_save_path, 'optimizer.pth'))

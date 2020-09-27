import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm # TODO : Remove TQDM
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models import *
from common import get_similar_traj

# NOTE : If in future, you operate on bigger hardwares - move to PyTorch Lightning
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def train_dummy_attn(config):
    tensorboard_writer = SummaryWriter(logdir=config.tensorboard_path)
    deg = config.deg()
    
    vobs_dim, dof_dim = deg.obs_space[deg.vis_obv_key], deg.obs_space[deg.dof_obv_key]
    act_dim = deg.action_space

    attn_module = BasicAttnModel(dof_dim, act_dim).to(device)
    params = list(attn_module.parameters())
    print("Number of parameters : {}".format(len(params)))
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    mse_loss = torch.nn.MSELoss()
    
    if(config.save_graphs):
        tensorboard_writer.add_graph(attn_module)
    if(config.resume):
        attn_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'attn_module.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(demons_config.models_save_path, 'optimizer.pth')))

    print("Run : `tensorboard --logdir={} --host '0.0.0.0' --port 6006`".format(config.tensorboard_path))
    if config.use_lang_search:
        dataloader = DataLoader(deg.instruct_dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)
    else:
        dataloader = deg.get_traj_dataloader(batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    max_step_size = len(dataloader.dataset)

    for epoch in tqdm(range(config.max_epochs), desc="Check Tensorboard"):
        if config.use_lang_search:
             for i, instruct_traj in enumerate(dataloader):
                instruction = instruct_traj['instruction']

                support_trajs = get_similar_traj(config, instruction)
                support_trajs = deg._collate_wrap()(support_trajs)
                support_trajs = {key : support_trajs[key].float().to(device) for key in support_trajs.keys()}
                
                key_dof_obs, key_actions = support_trajs[deg.dof_obv_key], support_trajs['action']
                key_batch_size, key_seq_len = key_dof_obs.shape[0], key_dof_obs.shape[1] 
                key_dof_obs = key_dof_obs.reshape(key_seq_len * key_batch_size, 1, -1)
                key_actions = key_actions.reshape(key_seq_len * key_batch_size, 1, -1)

                query_traj = {key : instruct_traj[key].float().to(device) for key in instruct_traj.keys() if key != 'instruction'}
                query_dof_obs, query_actions = query_traj[deg.dof_obv_key], query_traj['action']
                query_batch_size, query_seq_len = query_dof_obs.shape[0], query_dof_obs.shape[1] 
                query_dof_obs = query_dof_obs.reshape(query_seq_len, query_batch_size, -1)
                query_actions = query_actions.reshape(query_seq_len, query_batch_size, -1)

                preds = attn_module(curr_state=query_dof_obs, state_set=key_dof_obs, action_set=key_actions)

                loss = mse_loss(preds, query_actions)
                tensorboard_writer.add_scalar('lang_attn_loss', loss, epoch * max_step_size + i)
                
                loss.backward()
                optimizer.step()

                if int(i % config.save_interval_steps) == 0:
                    torch.save(attn_module.state_dict(), os.path.join(config.models_save_path, 'attn_module.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(config.models_save_path, 'optimizer.pth'))
        else:
            for i, trajectory in enumerate(dataloader):
                trajectory = {key : trajectory[key].float().to(device) for key in trajectory.keys()}
                dof_obs, actions = trajectory[deg.dof_obv_key], trajectory['action']
                batch_size, seq_len = dof_obs.shape[0], dof_obs.shape[1] 
                # NOTE : using ^ instead of config.batch_size coz diff. no. of samples possible from data. 

                dof_obs = dof_obs.reshape(seq_len, batch_size, -1)
                actions = actions.reshape(seq_len, batch_size, -1)
                state_set = dof_obs.reshape(seq_len * batch_size, 1, -1).repeat(1, batch_size, 1)
                action_set = actions.reshape(seq_len * batch_size, 1, -1).repeat(1, batch_size, 1)
                preds = attn_module(curr_state=dof_obs, state_set=state_set, action_set=action_set)

                optimizer.zero_grad()
                loss = mse_loss(preds, actions)
                tensorboard_writer.add_scalar('dummy_attn_loss', loss, epoch * max_step_size + i)
                
                loss.backward()
                optimizer.step()

                if int(i % config.save_interval_steps) == 0:
                    torch.save(attn_module.state_dict(), os.path.join(config.models_save_path, 'attn_module.pth'))
                    torch.save(optimizer.state_dict(), os.path.join(config.models_save_path, 'optimizer.pth'))

if __name__ == '__main__':
    from model_config import get_model_args
    config = get_model_args()
    torch.manual_seed(config.seed)
    
    train_dummy_attn(config)
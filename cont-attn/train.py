import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm # TODO : Remove TQDM
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models import *

# NOTE : If in future, you operate on bigger hardwares - move to PyTorch Lightning
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def train_dummy_attn(config):
    tensorboard_writer = SummaryWriter(logdir=config.tensorboard_path)
    deg = config.deg()
    
    vobs_dim, dof_dim = deg.obs_space[deg.vis_obv_key], deg.obs_space[deg.dof_obv_key] 
    act_dim = deg.action_space

    attn_module = BasicAttnModel(act_dim).to(device)
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
    data_loader = DataLoader(deg.traj_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    for epoch in tqdm(range(config.max_epochs), desc="Check Tensorboard"):
        for i, trajectory in enumerate(data_loader):
            trajectory = {key : trajectory[key].float().to(device) for key in trajectory.keys()}
            dof_obs, actions = trajectory[deg.dof_obv_key], trajectory['action']
            batch_size, seq_len = dof_obs.shape[0], dof_obs.shape[1] 
            # NOTE : using ^ instead of config.batch_size coz diff. no. of samples possible from data. 

            dof_obs = dof_obs.reshape(seq_len, batch_size, -1)
            actions = actions.reshape(seq_len, batch_size, -1)
            preds = attn_module(curr_state=dof_obs, state_set=dof_obs, action_set=actions)

            optimizer.zero_grad()
            loss = mse_loss(preds, actions)
            tensorboard_writer.add_scalar('dummy_attn_loss', loss)
            
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
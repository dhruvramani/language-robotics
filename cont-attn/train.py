import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
#from tqdm import tqdm # TODO : Remove TQDM
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from models import get_similar_traj

# NOTE : If in future, you operate on bigger hardwares - move to PyTorch Lightning
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(config):
    print("Device - ", device)
    tensorboard_writer = SummaryWriter(logdir=config.tensorboard_path)
    deg = config.deg()
    vobs_dim, dof_dim = deg.obs_space[deg.vis_obv_key], deg.obs_space[deg.dof_obv_key]
    act_dim = deg.action_space

    if config.use_visual_obv:
        percept = PerceptionModule(visual_obv_dim=vobs_dim, dof_obv_dim=dof_dim, state_dim=config.vis_emb_dim)
        dof_dim += config.vis_emb_dim

    if config.model == 'basic_attn':
        attn_module = BasicAttnModel(dof_dim, act_dim).to(device)
    elif config.model == 'rl_transformer':
        attn_module = RLTransformerEncoder(dof_dim, act_dim).to(device)

    params = list(attn_module.parameters())
    if config.use_visual_obv:
        params += list(percept.parameters())
    print("Number of parameters : {}".format(len(params)))
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    mse_loss = torch.nn.MSELoss()
    
    if(config.save_graphs):
        tensorboard_writer.add_graph(attn_module)
    if(config.resume):
        if config.use_visual_obv:
            percept.load_state_dict(torch.load(os.path.join(config.models_save_path, 'percept.pth')))
        attn_module.load_state_dict(torch.load(os.path.join(config.models_save_path, '{}.pth'.format(config.model))))
        optimizer.load_state_dict(torch.load(os.path.join(config.models_save_path, 'optimizer.pth')))

    print("Run : `tensorboard --logdir={} --host '0.0.0.0' --port 6006`".format(config.tensorboard_path))
    dataloader = deg.get_instruct_dataloader if config.use_lang_search else deg.get_traj_dataloader
    dataloader = dataloader(batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    max_step_size = len(dataloader.dataset)

    for epoch in range(config.max_epochs): #tqdm(, desc="Check Tensorboard"):
        if config.use_lang_search:
            loss_avg = 0.0
            for i, instruct_traj in enumerate(dataloader):
                support_trajs = get_similar_traj(config, deg, instruct_traj)
                support_trajs = {key : support_trajs[key].float().to(device) for key in support_trajs.keys()}
                
                key_dof_obs, key_actions = support_trajs[deg.dof_obv_key], support_trajs['action']
                key_batch_size, key_seq_len = key_dof_obs.shape[0], key_dof_obs.shape[1] 
                
                if config.use_visual_obv:
                    key_vis_obs = support_trajs[deg.vis_obv_key].reshape(key_seq_len * key_batch_size, -1)
                    key_vis_obs = key_vis_obs.reshape(-1, vobs_dim[2], vobs_dim[0], vobs_dim[1])
                    key_dof_obs = key_dof_obs.reshape(key_seq_len * key_batch_size, -1)
                    key_dof_obs = percept(key_vis_obs, key_dof_obs)

                key_dof_obs = key_dof_obs.reshape(key_seq_len, key_batch_size, -1)
                key_actions = key_actions.reshape(key_seq_len, key_batch_size, -1)

                query_traj = {key : instruct_traj[key].float().to(device) for key in instruct_traj.keys() 
                            if key in [deg.vis_obv_key, deg.dof_obv_key, 'action']}

                query_dof_obs, query_actions = query_traj[deg.dof_obv_key], query_traj['action']
                query_batch_size, query_seq_len = query_dof_obs.shape[0], query_dof_obs.shape[1] 

                if config.use_visual_obv:
                    query_vis_obs = query_traj[deg.vis_obv_key].reshape(query_seq_len * query_batch_size, -1)
                    query_vis_obs = query_vis_obs.reshape(-1, vobs_dim[2], vobs_dim[0], vobs_dim[1])
                    query_dof_obs = query_dof_obs.reshape(query_seq_len * query_batch_size, -1)
                    query_dof_obs = percept(query_vis_obs, query_dof_obs)

                query_dof_obs = query_dof_obs.reshape(query_seq_len, query_batch_size, -1)
                query_actions = query_actions.reshape(query_seq_len, query_batch_size, -1)

                # NOTE - Might have to debug here
                nopeak_mask = np.triu(np.ones((1, query_seq_len, query_seq_len)), k=1).astype('uint8')
                nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)

                preds = attn_module(curr_state=query_dof_obs, state_set=key_dof_obs, action_set=key_actions, mask=nopeak_mask)

                loss = mse_loss(preds, query_actions)
                loss_avg += loss
                tensorboard_writer.add_scalar('lang_{}_{}_loss'.format(config.model, "visual" if config.use_visual_obv else "state"), loss, epoch * max_step_size + i)
                
                loss.backward()
                optimizer.step()
                
                if int(i % config.save_interval_steps) == 0:
                    if config.use_visual_obv:
                        torch.save(percept.state_dict(), os.path.join(config.models_save_path, 'percept.pth'))
                    torch.save(attn_module.state_dict(), os.path.join(config.models_save_path, '{}.pth'.format(config.model)))
                    torch.save(optimizer.state_dict(), os.path.join(config.models_save_path, 'optimizer.pth'))
        else:
            # NOTE - This is for testing purposes only, remove in release.
            for i, trajectory in enumerate(dataloader):
                trajectory = {key : trajectory[key].float().to(device) for key in trajectory.keys()}
                dof_obs, actions = trajectory[deg.dof_obv_key], trajectory['action']
                batch_size, seq_len = dof_obs.shape[0], dof_obs.shape[1] 
                # NOTE : using ^ instead of config.batch_size coz diff. no. of samples possible from data. 

                if config.use_visual_obv:
                    vis_obs = trajectory[deg.key_vis_obs].reshape(seq_len * batch_size, -1)
                    dof_obs = dof_obs.reshape(seq_len * batch_size, -1)
                    dof_obs = percept(vis_obs, dof_obs)

                dof_obs = dof_obs.reshape(seq_len, batch_size, -1)
                actions = actions.reshape(seq_len, batch_size, -1)
                state_set = dof_obs.reshape(seq_len * batch_size, 1, -1).repeat(1, batch_size, 1)
                action_set = actions.reshape(seq_len * batch_size, 1, -1).repeat(1, batch_size, 1)
                preds = attn_module(curr_state=dof_obs, state_set=state_set, action_set=action_set)

                optimizer.zero_grad()
                loss = mse_loss(preds, actions)
                tensorboard_writer.add_scalar('{}_loss'.format(config.model), loss, epoch * max_step_size + i)
                
                loss.backward()
                optimizer.step()

                if int(i % config.save_interval_steps) == 0:
                    if config.use_visual_obv:
                        torch.save(percept.state_dict(), os.path.join(config.models_save_path, 'percept.pth'))
                    torch.save(attn_module.state_dict(), os.path.join(config.models_save_path, '{}.pth'.format(config.model)))
                    torch.save(optimizer.state_dict(), os.path.join(config.models_save_path, 'optimizer.pth'))

        print("Epoch {} | Loss : {}".format(epoch, loss_avg / len(dataloader.dataset)))


if __name__ == '__main__':
    from model_config import get_model_args
    config = get_model_args()
    torch.manual_seed(config.seed)
    
    train(config)
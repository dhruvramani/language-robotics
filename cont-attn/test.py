import os
import numpy
import torch
import matplotlib.pyplot as plt

from models import *
#from render_browser import render_browser

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def test_dummy(config):
    deg = config.deg()
    env = deg.get_env()
    vobs_dim, dof_dim = deg.obs_space[deg.vis_obv_key], deg.obs_space[deg.dof_obv_key] 
    act_dim = deg.action_space

    with torch.no_grad():
        attn_module = BasicAttnModel(dof_dim, act_dim).to(device)
        attn_module.load_state_dict(torch.load(os.path.join(config.models_save_path, 'attn_module.pth')))

        for i in range(config.n_test_evals):
            obvs, done, t = env.reset(), False, 0
            _set = next(iter(deg.get_traj_dataloader(batch_size=5)))
            state_set, action_set = _set[deg.dof_obv_key].float().to(device), _set['action'].float().to(device)
            batch_size, seq_len = state_set.shape[0], state_set.shape[1] 
            state_set = state_set.reshape(seq_len * batch_size, 1, -1)
            action_set = action_set.reshape(seq_len * batch_size, 1, -1)

            while (not done) and t <= config.max_test_timestep: 
                dof_obv = torch.from_numpy(deg._get_obs(obvs, deg.dof_obv_key)).float()
                dof_obv = dof_obv.reshape(1, 1, -1)
                
                action = attn_module(curr_state=dof_obv, state_set=state_set, action_set=action_set)
                obvs, _, _ = env.step(action[0, 0].numpy())
                plt.imsave("images/{}_{}.png".format(i, t), deg._get_obs(obvs, deg.vis_obv_key))
                print(t)
                t += 1
                
    deg.shutdown_env()

if __name__ == '__main__':
    from model_config import get_model_args
    config = get_model_args()
    test_dummy(config)
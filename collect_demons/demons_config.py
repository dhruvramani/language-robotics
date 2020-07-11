import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from global_config import *

def get_demons_args():
    parser = get_global_parser()
    parser.add_argument("--collect_by", type=str, default='play', choices=['play', 'imitation', 'expert', 'policy', 'exploration'])
    parser.add_argument("--device", type=str, default="keyboard", choices=["keyboard", "spacemouse"])
    parser.add_argument("--collect_freq", type=int, default=1)
    parser.add_argument("--flush_freq", type=int, default=75) # NOTE : RAM Issues, change here
    parser.add_argument("--break_traj_success", type=str2bool, default=True)
    parser.add_argument("--n_runs", type=int, default=10, 
        help="no. of runs of traj collection, affective when break_traj_success = False")

    # To store the model for imitating the play-data
    parser.add_argument('--train_imitation', type=str2bool, default=False)
    parser.add_argument('--tensorboard_path', type=str, default=os.path.join(BASE_DIR, 'runs/imitation-tensorboard/'))
    parser.add_argument('--load_models', type=str2bool, default=True)
    parser.add_argument('--use_model_perception_module', type=str2bool, default=True)
    parser.add_argument('--n_gen_traj', type=int, default=200, help="Number of trajectories to generate by imitation")

    config = parser.parse_args()
    config.data_path = os.path.join(config.data_path, '{}_{}/'.format(config.env, config.env_type)) 
    config.models_save_path = os.path.join(BASE_DIR, 'runs/imitation-models/{}_{}/'.format(config.env, config.env_type))
    config.tensorboard_path = os.path.join(config.tensorboard_path, '{}_{}_{}/'.format(config.env, config.env_type, config.exp_name)) 

    return config

import os
import sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '../'))

from global_config import *

def get_model_args():
    # TODO : Change w/ Hyperparams search later.
    parser = get_global_parser()

    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--is_train', type=str2bool, default=True)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--tensorboard_path', type=str, default=os.path.join(BASE_DIR, 'runs/tensorboard/'))
    parser.add_argument('--save_graphs', type=str2bool, default=False)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--n_test_evals', type=int, default=10)
    parser.add_argument('--max_test_timestep', type=int, default=40)

    parser.add_argument('--max_sequence_length', type=int, default=32)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--visual_state_dim', type=int, default=64)
    parser.add_argument('--combined_state_dim', type=int, default=72)
    parser.add_argument('--goal_dim', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=256)
    
    config = parser.parse_args()
    config.models_save_path = os.path.join(config.models_save_path, '{}-{}/'.format(config.env, config.exp_name)) 
    config.tensorboard_path = os.path.join(config.tensorboard_path, '{}-{}/'.format(config.env, config.exp_name)) 
    config.data_path = os.path.join(config.data_path, '{}/'.format(config.env)) 

    return config
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from global_config import *

def get_model_args():
    # TODO : Change w/ Hyperparams search later.
    parser = get_global_parser()

    # NOTE: Changed below. v
    parser.add_argument('--deg', type=env2deg, default='SURREAL')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--is_train', type=str2bool, default=True)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--tensorboard_path', type=str, default=os.path.join(BASE_DIR, 'runs/tensorboard/'))
    parser.add_argument('--save_graphs', type=str2bool, default=False)
    parser.add_argument('--save_interval_steps', type=int, default=100)
    parser.add_argument('--save_interval_epoch', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--max_steps_per_context', type=int, default=100)
    parser.add_argument('--context_steps_scale', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--n_test_evals', type=int, default=10)
    parser.add_argument('--max_test_timestep', type=int, default=40)

    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--visual_state_dim', type=int, default=64)
    parser.add_argument('--combined_state_dim', type=int, default=72)
    parser.add_argument('--goal_dim', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=256)

    parser.add_argument('--use_lang_model', type=str2bool, default=True)
    parser.add_argument('--lang_model', type=str, default='muse')
    
    config = parser.parse_args()
    config.deg = env2deg(config.env)
    if config.use_lang and not config.use_visual_models:
        config.models_save_path = os.path.join(config.models_save_path, '{}_{}_{}_{}/'.format(config.env, config.env_type, config.exp_name, config.lang_model)) 
        config.tensorboard_path = os.path.join(config.tensorboard_path, '{}_{}_{}_{}/'.format(config.env, config.env_type, config.exp_name, config.lang_model)) 
    else:
        config.models_save_path = os.path.join(config.models_save_path, '{}_{}_{}/'.format(config.env, config.env_type, config.exp_name)) 
        config.tensorboard_path = os.path.join(config.tensorboard_path, '{}_{}_{}/'.format(config.env, config.env_type, config.exp_name)) 
    config.data_path = os.path.join(config.data_path, '{}_{}/'.format(config.env, config.env_type)) 

    check_n_create_dir(config.models_save_path)
    check_n_create_dir(config.tensorboard_path)
    check_n_create_dir(config.data_path)

    return config
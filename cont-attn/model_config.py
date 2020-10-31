import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import utils
from global_config import *

def get_model_args():
    # TODO : Change w/ Hyperparams search later.
    parser = get_global_parser()

    # NOTE: 'SURREAL' is a placeholder. The deg is set according to global_config.env -> see below. v
    parser.add_argument('--deg', type=env2deg, default='SURREAL')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--is_train', type=utils.str2bool, default=False)
    parser.add_argument('--resume', type=utils.str2bool, default=False)

    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--save_graphs', type=utils.str2bool, default=False)
    parser.add_argument('--tensorboard_path', type=str, default=os.path.join(DATA_DIR, 'runs/tensorboard/'))
    parser.add_argument('--models_save_path', type=str, default=os.path.join(DATA_DIR, 'runs/models/'))
    parser.add_argument('--save_interval_steps', type=int, default=100)
    parser.add_argument('--save_interval_epoch', type=int, default=10)

    parser.add_argument('--max_cosine_cutoff', type=float, default=0.85)

    parser.add_argument('--max_epochs', type=int, default=75) #50
    parser.add_argument('--batch_size', type=int, default=1) # NOTE - Even if batch_size=1, seq_size ~ 200
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--n_test_evals', type=int, default=1)
    parser.add_argument('--max_test_timestep', type=int, default=150)

    parser.add_argument('--use_lang_search', type=utils.str2bool, default=True)
    parser.add_argument('--model', type=str, default='rl_transformer', choices=['basic_attn', 'rl_transformer', 'transformer_normal'])
    parser.add_argument('--lang_model', type=str, default='bert')
    parser.add_argument('--use_visual_obv', type=utils.str2bool, default=True)
    parser.add_argument('--vis_emb_dim', type=int, default=64)

    config = parser.parse_args()
    config.env_args = env2args(config.env)
    config.deg = env2deg(config.env)

    config.data_path = os.path.join(config.data_path, '{}_{}/'.format(config.env, config.env_type)) 
    if config.use_lang_search:
        config.models_save_path = os.path.join(config.models_save_path, 'contattn_{}_{}_{}_{}/'.format(config.env, config.env_type, config.exp_name, config.lang_model)) 
        config.tensorboard_path = os.path.join(config.tensorboard_path, 'contattn_{}_{}_{}_{}/'.format(config.env, config.env_type, config.exp_name, config.lang_model)) 
    else:
        config.models_save_path = os.path.join(config.models_save_path, 'contattn_{}_{}_{}/'.format(config.env, config.env_type, config.exp_name)) 
        config.tensorboard_path = os.path.join(config.tensorboard_path, 'contattn_{}_{}_{}/'.format(config.env, config.env_type, config.exp_name)) 

    if config.is_train and not config.resume:
        utils.recreate_dir(config.models_save_path, config.display_warnings)
        utils.recreate_dir(config.tensorboard_path, config.display_warnings)
    else:
        utils.check_n_create_dir(config.models_save_path, config.display_warnings)
        utils.check_n_create_dir(config.tensorboard_path, config.display_warnings)

    utils.check_n_create_dir(config.data_path, config.display_warnings)

    return config
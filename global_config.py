import os
import sys
import json
import datetime
import argparse

import utils

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_env/'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_env/surreal'))

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
DATA_DIR = '/scratch/scratch2/dhruvramani/language-robotics_data'
TIME_STAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# NOTE : Set it as true. Run `xvfb-run -a -s "-screen 0 1400x900x24" zsh` & 
#        `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so` for rendering.
def_env_args = dict(has_renderer=True, has_offscreen_renderer=True, ignore_done=True, use_camera_obs=True,  
    camera_height=256, camera_width=256, camera_name='agentview', use_object_obs=False, reward_shaping=True)
def_env_args = json.dumps(def_env_args)

def get_global_parser():
    ''' Global Config - contains global arguments common to all modules.            
        NOTE: All the paths/dirs have env-env_type-exp_name concated to them at the end. 
            - see model/model_config.py for example.
        TODO *IMPORTANT* : Stack 4-frames together while training and testing?
    '''

    parser = argparse.ArgumentParser("Language based robotics",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--env', type=str, default='SURREAL')
    parser.add_argument('--env_type', type=str, default='SawyerPickPlace')
    parser.add_argument('--env_args', type=json.loads, default=def_env_args)
    parser.add_argument('--exp_name', type=str, default='v0.5')
    parser.add_argument('--use_lang', type=utils.str2bool, default=False)
    parser.add_argument('--use_visual_models', type=utils.str2bool, default=True, 
        help='Load pretrained visual models for lang exps. See model_config.')
    parser.add_argument('--num_obv_types', type=int, default=2)
    parser.add_argument('--max_sequence_length', type=int, default=32)
    parser.add_argument('--data_path', type=str, default=os.path.join(DATA_DIR, 'data_files/saved_data/'))

    return parser

def env2deg(env_name):
    from dataset_env.surreal_deg import SurrealDataEnvGroup
    from dataset_env.furniture_deg import FurnitureDataEnvGroup

    # NOTE : *IMPORTANT* - Use these in model_config & demons_config ONLY.
    # *DEADLOCK* : can lead to deadlock if an object is created in global_config.
    env2deg_dict = {'SURREAL' : SurrealDataEnvGroup, 'FURNITURE' : FurnitureDataEnvGroup}

    if env_name is None:
        return None
    return env2deg_dict[env_name.upper()]

if __name__ == '__main__':
    print("=> Testing global_config.py")
    args = get_global_parser().parse_args()
    print(args.env)
    deg = env2deg('SURREAL')
    print(deg.action_space)
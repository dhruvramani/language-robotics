import os
import datetime
import argparse

def str2bool(string):
    return string.lower() == 'true'

def str2list(string):
    if not string:
        return []
    return [x for x in string.split(",")]

BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
TIME_STAMP = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def_env_args = dict(has_renderer=True, has_offscreen_renderer=True, ignore_done=True, use_camera_obs=True,  
    camera_height=84, camera_width=84, camera_name='agentview', use_object_obs=False, reward_shaping=True)
def_env_args = json.dumps(def_env_args)

def get_global_parser():
    ''' Global Config - contains global arguments common to all modules.            
        NOTE: All the paths/dirs have env-exp_name concated to them at the end. 
            - see model/model_config.py for example.
    '''

    parser = argparse.ArgumentParser("Language based robotics",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--env', type=str, default='SURREAL')
    parser.add_argument('--env_type', type=str, default='SawyerPickPlace')
    parser.add_argument('--env_args', type=json.loads, default=def_env_args)
    parser.add_argument('--exp_name', type=str, default='v0.2')
    parser.add_argument('--num_obv_types', type=int, default=2)
    parser.add_argument('--max_sequence_length', type=int, default=32)
    parser.add_argument('--models_save_path', type=str, default=os.path.join(BASE_DIR, 'runs/models/'))
    parser.add_argument('--data_path', type=str, default=os.path.join(BASE_DIR, 'data_files/saved_data/'))
    parser.add_argument('--use_lang', type=str2bool, default=False)


    return parser
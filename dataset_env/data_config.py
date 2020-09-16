import os
import sys
import argparse
import django

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../web_db/'))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_db.settings")
django.setup()

import utils
from global_config import *
from traj_db.models import SurrealRoboticsSuiteTrajectory, USCFurnitureTrajectory, RLBenchTrajectory
from hindsight_instruction.models import SurrealRoboticsSuiteInstruction, USCFurnitureInstruction, RLBenchInstruction

taj_db_dict = {'SURREAL' : SurrealRoboticsSuiteTrajectory, 'FURNITURE' : USCFurnitureTrajectory, 'RLBENCH' : RLBenchTrajectory}
instruct_db_dict = {'SURREAL' : SurrealRoboticsSuiteInstruction, 'FURNITURE' : USCFurnitureInstruction, 'RLBENCH' : RLBenchInstruction}

def env2TrajDB(string):
    if string is None:
        return None
    return taj_db_dict[string.upper()]

def env2InstructDB(string):
    if string is None:
        return None
    return instruct_db_dict[string.upper()]

def ep_type(string):
    ep_dict = {'teleop': 'EPISODE_ROBOT_PLAY', 'imitation': 'EPISODE_ROBOT_IMITATED',
                'expert': 'EPISODE_ROBOT_EXPERT', 'policy': 'EPISODE_ROBOT_POLICY',
                'exploration': 'EPISODE_ROBOT_EXPLORED', 'random': 'EPISODE_ROBOT_RANDOM'}
    return ep_dict[string.lower()]

def get_dataset_args():
    parser = get_global_parser()

    # NOTE: 'SURREAL' is a placeholder. The dbs are set according to global_config.env -> see below. v
    parser.add_argument('--traj_db', type=env2TrajDB, default='SURREAL')
    parser.add_argument('--instruct_db', type=env2InstructDB, default='SURREAL')
    parser.add_argument('--archives_path', type=str, default=os.path.join(DATA_DIR, 'data_files/archives'))
    parser.add_argument('--episode_type', type=ep_type, default='teleop', choices=['teleop', 'imitation', 'expert', 'policy', 'exploration', 'random'])
    parser.add_argument('--media_dir', type=str, default=os.path.join(BASE_DIR, 'web_db/static/media/'))
    parser.add_argument('--vid_path', type=str, default='vid.mp4')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--vocab_path', type=str, default=os.path.join(DATA_DIR, 'data_files/vocab.pkl'))

    parser.add_argument('--data_agumentation', type=utils.str2bool, default=False) # WARNING : Don't use now, UNSTABLE.
    parser.add_argument('--augs', type=utils.str2list, default='crop', help='See others in data_aug.py')
    
    config = parser.parse_args()
    config.env_args = env2args(config.env)
    config.traj_db = env2TrajDB(config.env)
    config.instruct_db = env2InstructDB(config.env)
    config.data_path = os.path.join(config.data_path, '{}_{}/'.format(config.env, config.env_type)) 
    config.archives_path = os.path.join(config.archives_path, '{}_{}/'.format(config.env, config.env_type)) 
    config.vid_path = os.path.join(config.media_dir, config.vid_path)

    utils.check_n_create_dir(config.data_path, config.display_warnings)
    utils.check_n_create_dir(config.archives_path, config.display_warnings)
    utils.check_n_create_dir(config.media_dir, config.display_warnings)

    return config

if __name__ == '__main__':
    print("=> Testing data_config.py")
    args = get_dataset_args()
    print(args.traj_db)
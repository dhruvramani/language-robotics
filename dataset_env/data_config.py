import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../web_db/traj_db/'))

from global_config import *
from models import SurrealRoboticsSuiteTrajectory, USCFurnitureTrajectory
from models import SurrealRoboticsSuiteInstruction, USCFurnitureInstruction

taj_db_dict = {'SURREAL' : SurrealRoboticsSuiteTrajectory, 'FURNITURE' : USCFurnitureTrajectory}
instruct_db_dict = {'SURREAL' : SurrealRoboticsSuiteInstruction, 'FURNITURE' : USCFurnitureInstruction}

def env2TrajDB(string):
    if string is None:
        return None
    return taj_db_dict[string.upper()]

def env2InstructDB(string):
    if string is None:
        return None
    return instruct_db_dict[string.upper()]

def ep_type(string):
    ep_dict = {'play': 'EPISODE_ROBOT_PLAY', 'imitation': 'EPISODE_ROBOT_IMITATED',
                'expert': 'EPISODE_ROBOT_EXPERT', 'policy': 'EPISODE_ROBOT_POLICY',
                'exploration': 'EPISODE_ROBOT_EXPLORED'}
    return ep_dict[string.lower()]

def get_dataset_args():
    parser = get_global_parser()

    # NOTE : Replaced by env_name below. v
    parser.add_argument('--traj_db', type=env2TrajDB, default='SURREAL')
    parser.add_argument('--instruct_db', type=env2InstructDB, default='SURREAL')
    parser.add_argument('--archives_path', type=str, default=os.path.join(BASE_DIR, 'data_files/archives'))
    parser.add_argument('--store_as', type=str, default='NumpyArray', choices=['TorchTensor', 'NumpyArray'])
    parser.add_argument('--episode_type', type=ep_type, default='play', choices=['play', 'imitation', 'expert', 'policy', 'exploration'])
    parser.add_argument('--media_dir', type=str, default=os.path.join(BASE_DIR, 'web_db/static/media/'))
    parser.add_argument('--vid_path', type=str, default='vid.mp4')
    parser.add_argument('--fps', type=int, default=30)
    
    config = parser.parse_args()
    config.traj_db = env2TrajDB(config.env)
    config.instruct_db = env2InstructDB(config.env)
    config.data_path = os.path.join(config.data_path, '{}_{}/'.format(config.env, config.env_type)) 
    config.archives_path = os.path.join(config.archives_path, '{}_{}/'.format(config.env, config.env_type)) 
    config.vid_path = os.path.join(config.media_dir, config.vid_path)

    check_n_create_dir(config.data_path)
    check_n_create_dir(config.archives_path)

    return config

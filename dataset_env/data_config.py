import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../web_db/traj_db/'))

from global_config import *
from models import SurrealRoboticsSuiteTrajectory, USCFurnitureTrajectory
from models import SurrealRoboticsSuiteInstruction, USCFurnitureInstruction

def env2TrajDB(string):
    if string is None:
        return None
    db_dict = {'SURREAL' : SurrealRoboticsSuiteTrajectory, 'FURNITURE' : USCFurnitureTrajectory}
    return db_dict[string.upper()]

def env2InstructDB(string):
    if string is None:
        return None
    db_dict = {'SURREAL' : SurrealRoboticsSuiteInstruction, 'FURNITURE' : USCFurnitureInstruction}
    return db_dict[string.upper()]

def ep_type(string):
    ep_dict = {'play': 'EPISODE_ROBOT_PLAY', 'imitation': 'EPISODE_ROBOT_IMITATED',
                'expert': 'EPISODE_ROBOT_EXPERT', 'policy': 'EPISODE_ROBOT_POLICY',
                'exploration': 'EPISODE_ROBOT_EXPLORED'}
    return ep_dict[string.lower()]

def get_dataset_args():
    parser = get_global_parser()

    # NOTE : SURREAL is a placeholder - replaced by env_name below.
    parser.add_argument('--traj_db', type=env2TrajDB, default='SURREAL')
    parser.add_argument('--instruct_db', type=env2InstructDB, default='SURREAL')
    parser.add_argument('--archives_path', type=str, default=os.path.join(BASE_DIR, 'data_files/archives'))
    parser.add_argument('--store_as', type=str, default='TorchTensor', choices=['TorchTensor', 'NumpyArray'])
    parser.add_argument('--episode_type', type=ep_type, default='play', choices=['play', 'imitation', 'expert', 'policy', 'exploration'])
    
    config = parser.parse_args()
    config.traj_db = env2TrajDB(config.env)
    config.instruct_db = env2InstructDB(config.env)
    config.data_path = os.path.join(config.data_path, '{}/'.format(config.env)) 
    config.archives_path = os.path.join(config.archives_path, '{}/'.format(config.env)) 

    return config

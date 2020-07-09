import os
import sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '../'))
sys.path.insert(1, os.path.join(sys.path[0], '../web_db/traj_db/'))

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

def get_dataset_args():
    parser = get_global_parser()

    # NOTE : SURREAL is a placeholder - replaced by env_name below.
    parser.add_argument('--traj_db', type=env2TrajDB, default='SURREAL')
    parser.add_argument('--instruct_db', type=env2InstructDB, default='SURREAL')
    parser.add_argument('--archives_path', type=str, default=os.path.join(BASE_DIR, 'data_files/archives'))
    parser.add_argument('--store_as', type=str, default='TorchTensor', choices=['TorchTensor', 'NumpyArray'])
    
    parser.add_argument('--episode_type', type=str, default='EPISODE_ROBOT_PLAY', 
        choices=['EPISODE_ROBOT_PLAY', 'EPISODE_ROBOT_IMITATED', 'EPISODE_ROBOT_EXPERT', 
        'EPISODE_ROBOT_POLICY', 'EPISODE_ROBOT_EXPLORED'])
    
    config = parser.parse_args()
    config.traj_db = env2TrajDB(config.env)
    config.instruct_db = env2InstructDB(config.env)
    config.data_path = os.path.join(config.data_path, '{}/'.format(config.env)) 
    config.archives_path = os.path.join(config.archives_path, '{}/'.format(config.env)) 

    return config

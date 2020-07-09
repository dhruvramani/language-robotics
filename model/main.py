import os
import numpy as np
import torch
import torch.nn.Functional as F

import utils
from model_config import get_model_args
from train import train_visual_goals
from test import test_experiment
from data_env import DataEnvGroup

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset_env'))

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def run():
    config = get_model_args()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.env == 'SURREAL':
        from surreal_deg import SurrealDataEnvGroup
        data_env_group = SurrealDataEnvGroup()
    elif config.env == 'FURNITURE':
        from furniture_deg import FurnitureDataEnvGroup
        data_env_group = FurnitureDataEnvGroup()

    if config.is_train:
        train_visual_goals(data_env_group, config)

    test_experiment(data_env_group, config)

if __name__ == '__main__':
    run()
import os
import numpy as np
import torch
import torch.nn.Functional as F

import utils
from model_config import get_model_args
from train import train_visual_goals
from test import test_experiment
from data_env import DataEnvGroup

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def run():
    config = get_model_args()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    data_env = DataEnvGroup(config.env)
    if config.is_train:
        dataset = data_env.get_dataset()
        train_visual_goals(data_env.get_env, dataset, config)

    test_experiment(data_env.get_env, config)

if __name__ == '__main__':
    run()
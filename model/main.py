import os
import numpy as np
import torch
import torch.nn.Functional as F

import utils
from model_config import get_model_args
from train import train_visual_goals
from test import test_experiment

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def run():
    config = get_model_args()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.is_train:
        train_visual_goals(config)

    test_experiment(config)

if __name__ == '__main__':
    run()
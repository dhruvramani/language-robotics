import os
import numpy as np
import torch
import torch.nn.functional as F

from model_config import get_model_args
from train import train_multi_context_goals
from test import test_experiment, test_with_lang

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def run():
    print("CHANGE exp_name TO NOT OVERRIDE PREV. EXPERIMENTS.")
    config = get_model_args()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.is_train:
        train_multi_context_goals(config)

    if config.use_lang:
        test_with_lang(config)
    else:
        test_experiment(config)

if __name__ == '__main__':
    run()
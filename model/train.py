import os
import torch
import numpy as np
import torch.nn.Functional as F

from models import *

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

def train_visual_goals(env_fn, dataset, config):

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    env, obs_dims, act_dim = env_fn()

    perception_module = PerceptionModule(obs_dims[0], obs_dims[1])
    goal_encoder = VisualGoalEncoder(perception_module)
    plan_recogonizer = PlanRecognizerModule()
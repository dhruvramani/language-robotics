import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

import data_aug as rad
from data_config import get_dataset_args
from file_storage import get_trajectory, get_random_trajectory, get_instruct_traj

class DataEnvGroup(object):
    ''' + NOTE : Create subclass for every environment, eg.
        Check `assert self.env_name == 'ENV_NAME'`
    '''
    def __init__(self, get_episode_type=None):
        ''' + Arguments:
                - get_episode_type: Get data of a particular episode_type (play, imitation, etc.)
                    > Default : None, get data with any episode_type.
        '''
        self.config = get_dataset_args()
        self.env_name = self.config.env
        self.env_type = self.config.env_type
        self.max_sequence_length = self.config.max_sequence_length
        self.episode_type = episode_type
        self.traj_dataset = self.TrajDataset(self.episode_type, self.config)
        self.instruct_dataset = self.InstructionDataset(self.config)

    def get_env(self):
        raise NotImplementedError

    def get_random_goal(self):
        goal_obs = get_random_trajectory()[0][-1, 0]
        goal = goal_obs[0]
        return goal

    class TrajDataset(Dataset):
        def __init__(self, episode_type, config):
            super(TrajDataset, self).__init__() # TODO : Might not be needed
            self.episode_type = episode_type
            self.config = config

        def __len__(self):
            if self.episode_type is None:
                return self.config.traj_db.objects.count()
            else:
                return self.config.traj_db.objects.filter(episode_type=self.episode_type).count()

        def __get_item__(self, idx):
            # NOTE : HUGE ASSUMPTION - assuming that the stored shapes are correct.
            # TODO : IMPORTANT - Implement trajectory cropping and all
            trajectory =  get_trajectory(index=idx, episode_type=self.episode_type)
            if self.config.data_agumentation:
                trajectory[:, 0] = rad.apply_augs(trajectory[:, 0], self.config)
            return trajectory

    class InstructionDataset(Dataset):
        def __init__(self, config):
            super(InstructionDataset, self).__init__()
            self.config = config

        def __len__(self):  
            self.config.instruct_db.objects.count()

        def __get_item__(self, idx):
            # TODO : IMPORTANT - Implement trajectory cropping and all
            instruction, traj = get_instruct_traj(index=idx)
             if self.config.data_agumentation:
                traj[:, 0] = rad.apply_augs(traj[:, 0], self.config)
            return instruction, traj


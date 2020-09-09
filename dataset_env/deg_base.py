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
        self.episode_type = get_episode_type
        self.traj_dataset = self.TrajDataset(self.episode_type, self.config)
        self.instruct_dataset = self.InstructionDataset(self.config)

        # NOTE : Environment dependent properties
        # Set these after inheriting the class. NotImplementedError
        self.vis_obv_key = None
        self.dof_obv_key = None
        self.obs_space = None
        self.action_space = None

    def get_env(self):
        raise NotImplementedError

    def teleoperate(self, demon_config):
        raise NotImplementedError

    def random_trajectory(self, demons_config):
        raise NotImplementedError

    def get_random_goal(self):
        assert issubclass(type(self), DataEnvGroup) is True # NOTE : might raise error - remove if so
        goal = get_random_trajectory()[0][self.vis_obv_key][-1]
        return goal

    class TrajDataset(Dataset):
        def __init__(self, episode_type, config):
            self.episode_type = episode_type
            self.config = config

        def __len__(self):
            if self.episode_type is None:
                return self.config.traj_db.objects.count()
            else:
                return self.config.traj_db.objects.filter(episode_type=self.episode_type).count()

        def __getitem__(self, idx):
            trajectory =  get_trajectory(index=idx, episode_type=self.episode_type)
            if self.config.data_agumentation:
                trajectory[self.vis_obv_key] = rad.apply_augs(trajectory[self.vis_obv_key], self.config)
            return trajectory

    class InstructionDataset(Dataset):
        def __init__(self, config):
            self.config = config

        def __len__(self):  
            return self.config.instruct_db.objects.count()

        def __getitem__(self, idx):
            instruction, trajectory = get_instruct_traj(index=idx)
            if self.config.data_agumentation:
                trajectory[self.vis_obv_key] = rad.apply_augs(trajectory[self.vis_obv_key], self.config)

            trajectory.update(instruction=instruction)
            return trajectory


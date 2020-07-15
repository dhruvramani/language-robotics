import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

from data_config import get_dataset_args
from file_storage import get_trajectory

class DataEnvGroup(object):
    ''' + NOTE : Create subclass for every environment, eg.
        Check `assert self.env_name == 'ENV_NAME'`
    '''
    def __init__(self, get_episode_type=None):
        ''' + Arguments:
                - get_episode_type: Get data of a particular episode_type. 
                    > Default : None, get data with any episode_type.
        '''
        self.config = get_dataset_args()
        self.env_name = self.config.env
        self.env_type = self.config.env_type
        self.max_sequence_length = self.config.max_sequence_length
        self.episode_type = episode_type
        self.dataset = self.TrajDataset(self.episode_type)

    def get_env(self):
        raise NotImplementedError

    def get_random_goal(self):
        goal_obs = get_trajectory(random=True)[-1, 0]
        goal = goal_obs[0]
        return goal

    class TrajDataset(Dataset):
        def __init__(self, episode_type):
            super(TrajDataset, self).__init__()
            self.episode_type = episode_type

        def __len__(self):
            if self.episode_type is None:
                return self.config.traj_db.objects.count()
            else:
                return self.config.traj_db.objects.filter(episode_type=self.episode_type).count()

        def __get_item__(self, idx):
            # NOTE : HUGE ASSUMPTION - assuming that the sotred shapes are correct.
            # TODO : IMPORTANT - Implement trajectory cropping and all
            trajectory =  get_trajectory(random=False, index=idx, episode_type=self.episode_type)
            # if self.config.store_as == 'NumpyArray':
            #     trajectory = torch.Tensor(trajectory)
            return trajectory
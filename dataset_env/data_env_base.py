import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

from data_config import get_dataset_args
from file_storage import get_trajectory

data_config = get_dataset_args()

class DataEnvGroup(object):
    ''' + NOTE : Create subclass for every environment, eg.

        class SurrealDataEnvGroup(DataEnvGroup):
            def __init__(self):
                super(SurrealDataEnvGroup, self).__init__()
                assert self.env_name == 'SURREAL' # NOTE - Most important
                ...

            # Override get_env method
            def get_env(self):
                ...
    '''
    def __init__(self):
        self.config = data_config
        self.env_name = self.config.env
        self.max_sequence_length = self.config.max_sequence_length
        self.dataset = TrajDataset()

    def get_env(self):
        raise NotImplementedError

    class TrajDataset(Dataset):
        def __len__(self):
            return self.config.traj_db.objects.count()

        def __get_item__(self, idx):
            # NOTE : HUGE ASSUMPTION - assuming that the sotred shapes are correct.
            trajectory =  get_trajectory(random=False, index=idx)
            if self.config.store_as == 'NumpyArray':
                trajectory = torch.Tensor(trajectory)
            return trajectory
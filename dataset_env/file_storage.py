import os
import sys
import torch
import tarfile
import numpy as np
from random import randint
from data_config import get_dataset_args, ep_type

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../web_db/traj_db/'))

from models import ArchiveFile

config = get_dataset_args()

def store_trajectoy(trajectory, episode_type=config.episode_type):
    ''' Arguments:
        - trajectory: [[Ot/st, at], ... K] where Ot = [Vobv, DOF, ...]
    '''
    if 'EPISODE_' not in episode_type:
        episode_type = ep_type(episode_type)

    if config.store_as == 'TorchTensor':
        trajectory = torch.Tensor(trajectory).cpu()
    elif config.store_as == 'NumpyArray' and type(trajectory) == torch.Tensor:
        trajectory = trajectory.cpu().numpy()

    assert trajectory.shape[-1] == 2 # Check if (obvs, actions)

    # NOTE : Current data_path is a placeholder. Edited below with UUID.
    metadata = config.traj_db(task_id=config.env_type, env_id=config.env, 
        data_path=config.data_path, episode_type=episode_type)

    metadata.save()
    metadata.data_path = os.path.join(config.data_path, "{}.pt".format(metadata.episode_id))
    metadata.save()

    if config.store_as == 'TorchTensor':
        torch.save(trajectory, metadata.data_path)
    elif config.store_as == 'NumpyArray':
        with open(metadata.data_path, 'wb') as file:
            np.save(file, trajectory)

def get_trajectory(random=True, index=None, episode_id=None):
    if random == False:
        assert episode_id is not None or index is not None
        if index is not None:
            metadata = config.traj_db.objects.get(task_id=config.env_type, traj_count=index)
        elif episode_id is not None:
            metadata = config.traj_db.objects.get(task_id=config.env_type, episode_id=episode_id)
    else :
        count = config.traj_db.objects.count()
        random_index = randint(1, count - 1)
        metadata = config.traj_db.objects.get(task_id=config.env_type, traj_count=random_index)
    
    trajectory = None
    if config.store_as == 'TorchTensor':
        trajectory = torch.load(metadata.data_path)
    elif onfig.store_as == 'NumpyArray':
        trajectory = np.load(metadata.data_path)

    return trajectory

def archive_traj_task(task=config.env_type, file_name=None):
    ''' 
        Archives trajectories by task (env_type)
        + Arguments:
            - task: config.env_type - group and archive them all
            - file_name: the name of the archive file. [NOTE: NOT THE PATH]
                >  NOTE : default file_name: `env_task.tar.gz`
    '''
    objects = config.traj_db.objects.get(task_id=task)
    
    if file_name is None:
        file_name = "{}_{}.tar.gz".format(config.env, config.env_type)
    file_name = os.path.join(config.archives_path, file_name)

    tar = tarfile.open(file_name, "w:gz")
    for metadata in objects:
        if metadata.is_archived == False:
            metadata.is_archived = True
            metadata.save()
        
            tar.add(metadata.data_path)

            archive = ArchiveFile(trajectory=metadata, env_id=metadata.env_id, archive_file=file_name)        
            archive.save()

    tar.close()
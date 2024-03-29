import re
import os
import sys
import uuid
import torch
import pickle
import tarfile
import numpy as np
from random import randint
from data_config import get_dataset_args, ep_type

from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../traj_db/'))
config = get_dataset_args()

from traj_db.models import ArchiveFile

def store_trajectoy(trajectory, episode_type=config.episode_type, task=None):
    ''' 
        Save trajectory to the corresponding database based on env and env_type specified in config.
        + Arguments:
            - trajectory: {deg.vis_obv_key : np.array([n]), deg.dof_obv_key : np.array([n]), 'action' : np.array([n])}
            - episode_type [optional]: tag to store trajectories with (eg. 'teleop' or 'imitation')
    '''
    if 'EPISODE_' not in episode_type:
        episode_type = ep_type(episode_type)
    if task is None:
        task = config.env_type
    assert 'action' in trajectory.keys()

    # NOTE : Current data_path is a placeholder. Edited below with UUID.
    metadata = config.traj_db(task_id=task, env_id=config.env, 
        data_path=config.data_path, episode_type=episode_type, traj_steps=trajectory['action'].shape[0])

    metadata.save()
    metadata.data_path = os.path.join(config.data_path, "traj_{}.pt".format(metadata.episode_id))
    metadata.save()

    with open(metadata.data_path, 'wb') as file:
        pickle.dump(trajectory, file, protocol=pickle.HIGHEST_PROTOCOL)

def save_instruct_traj(traj_id, instruction):
    import common

    traj_id = str(traj_id)
    trajectory = config.traj_db.objects.get(episode_id=uuid.UUID(traj_id))
    instruct_obj = config.instruct_db(env_id=trajectory.env_id, task_id=trajectory.task_id,
        instruction=instruction, trajectory=trajectory, instruction_path=config.data_path)
    instruct_obj.save()
    instruct_obj.instruction_path = os.path.join(config.data_path, "instruct_{}_{}.pt".format(instruct_obj.instruction_id, traj_id))
    instruct_obj.save()

    lang_model = common.LanguageModelInstructionEncoder('bert')
    instruct_dict = lang_model(instruction)
    instruct_dict.update(instruction=instruction)

    with open(instruct_obj.instruction_path, 'wb') as file:
        pickle.dump(instruct_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    return True

def get_instruct_traj(index=None, instruction_id=None):
    '''
        Gets a particular instruction & corresponding trajectory from the corresponding database based on env and env_type specified in config.
        + Arguments:
            - index [optional]: get instruction object at a particular index
            - instruction_id [optional]: get instruction object by it's instruction_id (primary key)
          
        + NOTE: either index or episode_id should be not None.
    '''
    if index is None and instruction_id is None:
        return [get_instruct_traj(instruction_id=instruct_obj.instruction_id) for instruct_obj in config.instruct_db.objects.all()]

    if index is not None:
        instruction_obj = config.instruct_db.objects.filter(task_id=config.env_type)[index] if config.get_by_task_id else config.instruct_db.objects.all()[index]
    elif instruction_id is not None:
        instruction_id = str(instruction_id)
        instruction_obj = config.instruct_db.objects.get(instruction_id=uuid.UUID(instruction_id))

    with open(instruction_obj.instruction_path, 'rb') as file:
        instruct_dict = pickle.load(file)

    trajectory_obj = instruction_obj.trajectory
    trajectory = get_trajectory(episode_id=str(trajectory_obj.episode_id))
    return instruct_dict, trajectory

def get_trajectory(episode_type=None, index=None, episode_id=None):
    '''
        Gets a particular trajectory from the corresponding database based on env and env_type specified in config.
        + Arguments:
            - episode_type [optional]: if you want trajectory specific to one episode_type (eg. 'teleop' or 'imitation')
            - index [optional]: get trajectory at a particular index
            - episode_id [optional]: get trajectory by it's episode_id (primary key)
        
        + NOTE: either index or episode_id should be not None.
        + NOTE: episode_type, env_type become POINTLESS when you pass episode_id.
    '''
    # TODO : If trajectory is in archive-file, get it from there
    if episode_id is None and index is None:
        return [get_trajectory(episode_id=traj_obj.episode_id) for traj_obj in config.traj_db.objects.all()]

    if index is not None:
        if episode_type is None: # TODO : Clean code
            metadata = config.traj_db.objects.filter(task_id=config.env_type)[index] if config.get_by_task_id else config.traj_db.objects.all()[index]
        else:
            metadata = config.traj_db.objects.filter(task_id=config.env_type, episode_type=episode_type)[index] if config.get_by_task_id else config.traj_db.objects.filter(episode_type=episode_type)[index]
    elif episode_id is not None:
        episode_id = str(episode_id)
        metadata = config.traj_db.objects.get(episode_id=uuid.UUID(episode_id))

    with open(metadata.data_path, 'rb') as file:
        trajectory = pickle.load(file)

    return trajectory

def get_instruct_traj_non_db(index=None, instruction_id=None):
    files = [file for file in os.listdir(config.data_path) if 'instruct' in file]
    if instruction_id is None and index is None:
        return [get_trajectory_non_db(index=i) for i in range(len(files))]

    if index is not None:
        file_path = config.data_path + files[index]
    elif instruction_id is not None:
        file_path = config.data_path + [file for file in files if instruction_id in file][0]

    with open(file_path, 'rb') as file:
        instruct_dict = pickle.load(file)

    traj_id = file_path.split('/')[-1].split('_')[-1].split('.')[0]
    trajectory = get_trajectory_non_db(episode_id=traj_id)

    return instruct_dict, trajectory

def get_trajectory_non_db(index=None, episode_id=None):
    files = [file for file in os.listdir(config.data_path) if 'traj' in file]
    if episode_id is None and index is None:
        return [get_trajectory_non_db(index=i) for i in range(len(files))]

    if index is not None:
        file_path = config.data_path + files[index]
    elif episode_id is not None:
        file_path = config.data_path + [file for file in files if episode_id in file][0]

    with open(file_path, 'rb') as file:
        trajectory = pickle.load(file)

    return trajectory
    
def get_random_trajectory(episode_type=None):
    '''
        Gets a random trajectory from the corresponding database based on env and env_type specified in config.
        + Arguments:
            - episode_type [optional]: if you want trajectory specific to one episode_type (eg. 'teleop' or 'imitation')
    '''
    count = config.traj_db.objects.count()
    random_index = randint(1, count)
    if episode_type is None:
        metadata = config.traj_db.objects.filter(task_id=config.env_type)[random_index] if config.get_by_task_id else config.traj_db.objects.all()[random_index]
    else:
        metadata = config.traj_db.objects.filter(task_id=config.env_type, episode_type=episode_type)[random_index] if config.get_by_task_id else config.traj_db.objects.filter(episode_type=episode_type)[random_index]
    
    episode_id = str(metadata.episode_id)
    task_id = metadata.task_id
    trajectory = get_trajectory(episode_id=episode_id)

    return trajectory, episode_id, task_id

def create_video(trajectory):
    '''
        Creates videos and stores video, the initial and the final frame in the paths specified in data_config. 
        + Arguments:
            - trajectory: {deg.vis_obv_key : np.array([n]), deg.dof_obv_key : np.array([n]), 'action' : np.array([n])}
    '''
    import torchvision

    frames = trajectory[config.obv_keys['vis_obv_key']].astype(np.uint8)
    assert frames.shape[-1] == 3
    
    inital_obv, goal_obv = Image.fromarray(frames[0]), Image.fromarray(frames[-1])
    inital_obv.save(os.path.join(config.media_dir, 'inital.png'))
    goal_obv.save(os.path.join(config.media_dir, 'goal.png'))

    if type(frames) is not torch.Tensor:
        frames = torch.from_numpy(frames)

    torchvision.io.write_video(config.vid_path, frames, config.fps)
    return config.vid_path

# NOT TESTED
def archive_traj_task(task=config.env_type, episode_type=None, file_name=None):
    ''' 
        Archives trajectories by task (env_type)
        + Arguments:
            - task: config.env_type - group and archive them all.
            - episode_type [optional]: store trajectories w/ same task, episode_type together.
            - file_name: the name of the archive file. [NOTE: NOT THE PATH]
                >  NOTE : default file_name: `env_task.tar.gz`
    '''    
    if episode_type is None:
        objects = config.traj_db.objects.get(task_id=task)
        f_name = "{}_{}.tar.gz".format(config.env, config.env_type)
    else:
        objects = config.traj_db.objects.get(task_id=task, episode_type=episode_type)
        f_name = "{}_{}_{}.tar.gz".format(config.env, config.env_type, episode_type)
    
    if file_name is None:
        file_name = f_name
    file_name = os.path.join(config.archives_path, file_name)

    tar = tarfile.open(file_name, "w:gz")
    for metadata in objects:
        if metadata.is_archived == True:
            continue
            
        metadata.is_archived = True
        metadata.save()
        
        tar.add(metadata.data_path)

        archive = ArchiveFile(trajectory=metadata, env_id=metadata.env_id, archive_file=file_name)        
        archive.save()
    tar.close()

def delete_trajectory(episode_id):
    obj = config.traj_db.objects.get(episode_id=uuid.UUID(episode_id))
    if os.path.exists(obj.data_path):
        os.remove(obj.data_path)
    obj.delete()

def delete_instruct(instruction_id):
    obj = config.instruct_db.objects.get(instruction_id=uuid.UUID(instruction_id))
    if os.path.exists(obj.instruction_path):
        os.remove(obj.instruction_path)
    obj.delete()

def flush_traj_db():
    raise NotImplementedError

def get_vocab2idx():
    vocab = []
    if os.path.isfile(config.vocab_path):
        with open(config.vocab_path, 'rb') as pkl_file:
            vocab = pickle.load(pickle_file)
    vocab = {word : i for i, word in enumerate(vocab)}
    return vocab

def add_vocab(sentence):
    vocab = []
    if os.path.isfile(config.vocab_path):
        with open(config.vocab_path, 'rb') as pkl_file:
            vocab = pickle.load(pickle_file)
    vocab = set(list(vocab) + re.sub("[^\w]", " ",  sentence).split())
    with open(config.vocab_path, 'wb') as pkl_file:
        pickle.dump(vocab, pkl_file)

def rename_instruct_trajs():
    for instruct_obj in config.instruct_db.objects.all():
        trajectory_obj = instruct_obj.trajectory
        new_path = os.path.join(config.data_path, "instruct_{}_{}.pt".format(instruct_obj.instruction_id, str(trajectory_obj.episode_id)))
        if os.path.isfile(instruct_obj.instruction_path):
            os.rename(instruct_obj.instruction_path, new_path)
        instruct_obj.instruction_path = new_path

if __name__ == '__main__':
    print(len(config.instruct_db.objects.all()))
    # files = [file for file in os.listdir(config.data_path) if 'instruct' in file]
    # for instruct_obj in config.instruct_db.objects.all():
    #     file_ids = [f.split('_')[1].split('.')[0] for f in files]
    #     if instruct_obj.instruction_id not in file_ids:
    #         print(instruct_obj.instruction_id)
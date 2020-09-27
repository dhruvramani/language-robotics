import os
import sys
import uuid
from django.shortcuts import render

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../dataset_env'))

from data_config import get_dataset_args
import file_storage

config = get_dataset_args()

def hindsight_instruction(request):
    save_success = False
    if request.method == 'POST':
        episode_id = uuid.UUID(request.POST['episode_id'])
        instruction = request.POST['instruction'].lower()
        #file_storage.add_vocab(instruction)
        save_success = file_storage.save_instruct_traj(episode_id, instruction)

    trajectory, episode_id = file_storage.get_random_trajectory()
    vid_path = file_storage.create_video(trajectory)
    assert os.path.isfile(vid_path)

    return render(request, 'h_instruction.html', {'episode_id': episode_id, 'save_success': save_success})

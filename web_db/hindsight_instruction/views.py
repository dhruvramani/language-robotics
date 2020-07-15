import os
import sys
import uuid
from django.shortcuts import render
from .forms import InstructionForm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../dataset_env'))

from data_config import get_dataset_args
from file_storage import get_random_trajectory, create_video

config = get_dataset_args()

def hindsight_instruction(request):
    save_success = False
    if request.method == 'POST':
        instruction_form = InstructionForm(request.POST)
        
        if instruction_form.is_valid():
            episode_id = uuid.UUID(instruction_form.cleaned_data['episode_id'])
            instruction = instruction_form.cleaned_data['instruction']

            trajectory = config.traj_db.objects.get(episode_id=episode_id)
            instruct_obj = config.instruct_db(env_id=trajectory.env_id, user=request.user, 
                instruction=instruction, trajectory=trajectory)
            instruct_obj.save()
            save_success = True

    trajectory, episode_id = get_random_trajectory()
    vid_path = create_video(trajectory)
    assert os.path.isfile(vid_path)

    return render(request, 'h_instruction.html', {'episode_id': episode_id, 'save_success': save_success})

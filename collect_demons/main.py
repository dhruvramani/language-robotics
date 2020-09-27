import os 
import sys
import numpy as np
import torch

from demons_config import get_demons_args

if __name__ == "__main__":
    demon_config = get_demons_args()
    deg = demon_config.deg()
    if demon_config.collect_by == 'teleop':
        deg.teleoperate(demon_config)
    elif demon_config.collect_by == 'random':
        deg.random_trajectory(demon_config)
    elif demon_config.collect_by == 'imitation':
        #NOTE : NOT TESTED
        import imitate_play
        
        if demon_config.train_imitation:
            imitate_play.train_imitation(demon_config)
        imitate_play.imitate_play()
    elif demon_config.collect_by == 'instruction':
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset_env'))
        import file_storage

        _, episode_id = file_storage.get_random_trajectory()
        instruction = input("Enter instruction : ")
        save_success = file_storage.save_instruct_traj(episode_id, instruction)
        if save_success:
            print("Instruction Saved")

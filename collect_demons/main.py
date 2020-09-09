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
    elif demons_config.collect_by == 'imitation':
        # NOT TESTED
        import imitate_play
        
        if demons_config.train_imitation:
            imitate_play.train_imitation(demons_config)
        imitate_play.imitate_play()
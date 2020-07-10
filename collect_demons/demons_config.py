import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from global_config import *

def get_demons_args():
    parser = get_global_parser()
    parser.add_argument("--device", type=str, default="keyboard", choices=["keyboard", "spacemouse"])
    parser.add_argument("--collect_freq", type=int, default=1)
    parser.add_argument("--flush_freq", type=int, default=75) # NOTE : RAM Issues, change here
    parser.add_argument("--break_traj_success", type=str2bool, default=True)
    parser.add_argument("--n_runs", type=int, default=10, 
        help="number of runs of traj collection, affective when break_traj_success = False")

    config = parser.parse_args()
    config.data_path = os.path.join(config.data_path, '{}_{}/'.format(config.env, config.env_type)) 

    return config

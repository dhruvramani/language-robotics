import os
import argparse

def str2bool(string):
    return string.lower() == 'true'

def str2list(string):
    if not string:
        return []
    return [x for x in string.split(",")]

def get_model_args(parser=None):
    # TODO : Change w/ Hyperparams search later.

    isParser = True
    if parser is None : 
        isParser = False
        parser = argparse.ArgumentParser("Model for language based robotics",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # TODO
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='v0.1')
    parser.add_argument('--save_path', type='str', './runs/models')
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--save_graphs', type=str2bool, default=False)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)

    parser.add_argument('--env', type=str, default='')
    parser.add_argument('--use_lang', type=str2bool, default=False)
    parser.add_argument('--max_sequence_length', type=int, default=32)
    parser.add_argument('--num_obv_types', type=int, default=2)
    parser.add_argument('--beta', type=float, default=0.01)

    if isParser:
        return parser

    return parser.parse_args()
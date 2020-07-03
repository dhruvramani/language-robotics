import os
import argparse

def str2bool(string):
    return string.lower() == 'true'

def str2list(string):
    if not string:
        return []
    return [x for x in string.split(",")]

def get_model_args(parser=None):
    
    isParser = True
    if parser is None : 
        isParser = False
        parser = argparse.ArgumentParser("Model for language based robotics",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # TODO
    parser.add_argument('--env', type=str, default='')

    parser.add_argument('--exp_name', type=str, default='v0.1')
    parser.add_argument('--use_lang', type=str2bool, default=False)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from checkpoint')
    parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from checkpoint')


    
    # TODO : Change w/ Hyperparams search later
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_iters', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument('--max_sequence_length', type=int, default=32)

    if isParser:
        return parser

    return parser.parse_args()
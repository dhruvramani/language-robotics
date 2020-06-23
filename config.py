import os
import argparse
from model.config import get_model_args

def str2bool(string):
    return string.lower() == 'true'

def str2list(string):
    if not string:
        return []
    return [x for x in string.split(",")]

def argparser():
    parser = argparse.ArgumentParser("Language based robotics",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser = get_model_args(parser)

    #TODO - WebApp Arguments
    parser.add_argument('--')

    return parser.parse_args()
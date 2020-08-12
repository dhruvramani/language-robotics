import os
import sys
import pathlib

def check_n_create_dir(path):
    if not os.path.isdir(path):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    else:
        print("=> Directory {} exists (no new directory created).".format(path))

def str2bool(string):
    return string.lower() == 'true'

def str2list(string):
    if not string:
        return []
    return [x for x in string.split(",")]

import os
import json
from hashlib import md5
from time import localtime

def list_files(dir):
    return [os.path.join(dir, p.name) for p in os.scandir(dir)]

def list_dirs(dir):
    return [p.name for p in os.scandir(dir) if os.path.isdir(os.path.join(dir, p.name))]

def get_filename(p):
    return os.path.basename(p)


def get_filename_without_ext(p):
    return os.path.splitext(os.path.basename(p))[0]


def get_filename_and_ext(p):
    return os.path.splitext(os.path.basename(p))


def add_prefix_to(filename):
    prefix = md5(str(localtime()).encode('utf-8')).hexdigest()
    return f"{prefix}_{filename}"


def write_json(file_path, data):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def load_json(file_path):
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
        return json_data
    return None

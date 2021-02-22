
import os
import json
import time
from hashlib import md5
from time import localtime


def list_entries(dir):
    if not os.path.exists(dir):
        return []
    return [os.path.join(dir, p.name) for p in os.scandir(dir)]


def list_dirs(dir):
    if not os.path.exists(dir):
        return []
    return [p.name for p in os.scandir(dir) if os.path.isdir(os.path.join(dir, p.name))]


def list_files(dir):
    if not os.path.exists(dir):
        return []
    return [os.path.join(dir, p.name) for p in os.scandir(dir) if os.path.isfile(os.path.join(dir, p.name))]


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
    with open(file_path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
        return json_data
    return None


def wait_until_generated(file_path, wait_sec=1, retry_count=10):
    time_counter = 0
    while not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        time.sleep(wait_sec)
        time_counter += 1
        if time_counter > retry_count:
            break

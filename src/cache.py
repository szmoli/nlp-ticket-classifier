import os
import pickle
import hashlib

def file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def cache_exists(path):
    return os.path.exists(path)

def save_cache(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_cache(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
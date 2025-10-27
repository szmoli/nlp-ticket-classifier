import os
import pickle
import hashlib

def hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def exists(path):
    return os.path.exists(path)

def save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
import random
import pickle
import GPUtil
import numpy as np
import base64
from datasets import load_dataset
import torch
import pathlib

# Create data batches
def batchify(lst, batch_size):
    """Yield successive batches from list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def shuffle(inputs, labels):
    # Generate a random permutation of indices based on the number of samples
    indices = torch.randperm(inputs.size(0))

    # Shuffle data and labels using the generated indices
    shuffled_data = inputs[indices]
    shuffled_labels = labels[indices]
    return shuffled_data, shuffled_labels


def save_dict(data, dir):
    with open(dir, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_dict(dir):
    with open(dir, 'rb') as fp:
        data = pickle.load(fp)
    return data

def create_dir(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True) 
    
def get_free_gpu():
    # Get a list of all GPUs and their status
    gpus = GPUtil.getGPUs()
    # If there are no GPUs available, raise an error
    if not gpus:
        raise RuntimeError("No GPU available.")
    # Sort GPUs by available memory (descending order)
    gpus_sorted_by_memory = sorted(gpus, key=lambda gpu: gpu.memoryFree, reverse=True)
    # Select the GPU with the most free memory
    selected_gpu = gpus_sorted_by_memory[0]
    print(f"Selected GPU ID: {selected_gpu.id}")
    return selected_gpu.id
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

def load_datasets(malicious_only=False):
    # Load datasets
    all_texts = []
    all_labels = []
    
    # ds = load_dataset("walledai/MaliciousInstruct")
    # all_texts += ds['train']['prompt']
    # all_labels += [1] * len(ds['train']['prompt'])

    # ds = load_dataset("walledai/StrongREJECT")
    # all_texts += ds['train']['prompt']
    # all_labels += [1] * len(ds['train']['prompt'])

    # ds = load_dataset("walledai/TDC23-RedTeaming")
    # all_texts += ds['train']['prompt']
    # all_labels += [1] * len(ds['train']['prompt'])
    
    ds = load_dataset("walledai/CatHarmfulQA")
    all_texts += ds['en']['prompt'] #en/ch/ve
    all_labels += [1] * len(ds['en']['prompt'])
    
    ds = load_dataset("declare-lab/HarmfulQA")
    all_texts += ds['train']['question']
    all_labels += [1] * len(ds['train']['question'])

    ds = load_dataset("LLM-LAT/harmful-dataset")
    all_texts += ds['train']['prompt']
    all_labels += [1] * len(ds['train']['prompt'])

    print(f'Number of malicious promts: {len(all_texts)}')
    
    if not malicious_only:
        # Benign dataset
        ds = load_dataset("facebook/natural_reasoning")
        all_texts += ds['train']['question'][:len(all_labels)]
        all_labels += [0] * len(all_labels)
    
        print(f'Number of all promts: {len(all_texts)}')
    
    return all_texts, all_labels

def load_code_datasets():
    # Load datasets
    all_texts = []
    all_labels = []

    ds = load_dataset("CyberNative/Code_Vulnerability_Security_DPO")
    all_texts += ds['train']['chosen']
    all_labels += [0] * len(ds['train']['chosen'])
    all_texts += ds['train']['rejected']
    all_labels += [1] * len(ds['train']['rejected'])
    return all_texts, all_labels

def expand_data(prompts, labels, num_responses=10):
    """
    Given a list of prompts and their labels, expand them so that each prompt
    is repeated num_responses times, and the corresponding label is also repeated.
    """
    expanded_labels = []
    expanded_prompts = []
    for prompt, label in zip(prompts, labels):
        expanded_prompts.extend([prompt] * num_responses)
        expanded_labels.extend([label] * num_responses)
    return expanded_prompts, expanded_labels

# def shuffle(inputs, labels):
#     # Create a list of (text, label) pairs and shuffle
#     data = list(zip(inputs, labels))
#     random.shuffle(data)

#     # Unzip shuffled data
#     shuffled_texts, shuffled_labels = zip(*data)
#     return shuffled_texts, shuffled_labels

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
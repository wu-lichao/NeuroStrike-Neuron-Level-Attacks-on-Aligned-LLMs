import numpy as np
from datasets import load_dataset
import torch

import probe
import util
import util_model
from activation_extractor import NeuronActivationExtractor

def flatten_activation(query_activations):
    act_list = []
    for i, (layer_name, act) in enumerate(query_activations.items()):
        if isinstance(act, torch.Tensor):
            act = act.detach().cpu().numpy()  # Ensure tensor is detached and on CPU
        act_list.append(act)
    activations = np.concatenate(act_list, axis=1)
    return activations

if __name__ == "__main__":
    # Select the model that you want to test
    model_id = 0
    
    # auto: use all gpu
    # cpu: use cpu only
    device = 'auto'
    
    safe_neuron_threshold = 3
    batch_size = 32
    num_queries = 10 # change this value to adapte the training set size
    
    model_list = [
        "google/gemma-3-1b-it", # 0
        "Qwen/Qwen2.5-32B-Instruct", # 1
    ]

    model_name = model_list[model_id].split('/')[1]
    print(f"Evaluating {model_name}")

    # Load the tokenizer and model configuration
    model, tokenizer = util_model.load_model(model_list[model_id], device)

    # Instantiate the extractor. Optionally, you can pass target_layers or let it auto-detect.
    extractor = NeuronActivationExtractor(
        model, tokenizer, model_name,
        safe_neuron_threshold=safe_neuron_threshold
    )
    device = model.device

    jailbreakv_28k_ds = load_dataset("JailbreakV-28K/JailBreakV-28k", 'JailBreakV_28K')["JailBreakV_28K"]
    jailbreakv_28k_ds = jailbreakv_28k_ds.filter(lambda ex: ex["format"] == "Template")    
    redteam_query = jailbreakv_28k_ds['redteam_query']
    jailbreak_query = jailbreakv_28k_ds['jailbreak_query']
    
    ds = load_dataset("facebook/natural_reasoning")
    benign_query = ds['train']['question'][:len(redteam_query)+len(jailbreak_query)]
    
    benign_query = benign_query[:num_queries*2]
    redteam_query = redteam_query[:num_queries]
    jailbreak_query = jailbreak_query[:num_queries]
    
    print(len(benign_query), len(redteam_query), len(jailbreak_query))
    query_dataset = np.concatenate([benign_query, redteam_query, jailbreak_query])
    query_activations = extractor.get_activations(query_dataset, batch_size=batch_size)

    # get labels
    labels = []
    labels += [1]*len(benign_query)
    labels += [0]*len(redteam_query)
    labels += [0]*len(jailbreak_query)
    activations = flatten_activation(query_activations)
    print(np.shape(activations))
    
    # Perform the safety probe
    clf = probe.safety_probe(activations, labels, device=device, num_runs=1)
    scripted_model = torch.jit.script(clf)
    util.create_dir("./_scorer") 
    scripted_model.save(f'./_scorer/scorer_{model_name}.pt')
    
    # Test saved moodel
    ensemble_model = torch.jit.load(f'./_scorer/scorer_{model_name}.pt').to(model.device).eval()
    pred = ensemble_model.predict(torch.tensor(activations, dtype=torch.float32).to(model.device))
    print(pred.size())
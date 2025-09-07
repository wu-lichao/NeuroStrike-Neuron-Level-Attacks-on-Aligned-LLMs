from scipy.stats import zscore
import numpy as np
from tqdm import tqdm
import re
import torch
import pickle
import util
import util_model

class NeuronActivationExtractor:
    def __init__(self, model, tokenizer, model_name, safe_neuron_threshold=3, scorer_dir=None, target_layers=None, get_activation=True):
        """
        Initialize the extractor with the given model, tokenizer, file root, model name,
        and safe neuron threshold. If target_layers is not provided, it will auto-detect 
        the number of MLP layers via util.count_mlp_module.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.safe_neuron_threshold = safe_neuron_threshold
        if scorer_dir is not None:
            torch.jit.load(scorer_dir, map_location=self.model.device).to(self.model.device).eval()
        self.get_activation = get_activation
        
        # Compute candidate (safety) neurons from pre-saved weights.
        self.candidate_dict = self.get_safety_neurons()
        
        # Determine target layers. If not provided, select all mlp layers.
        if target_layers is None:
            num_mlp = self.count_mlp_module(self.model, self.model_name)
            self.target_layers = list(range(num_mlp))
        else:
            self.target_layers = target_layers

        if self.get_activation:
            self.activations = {}  # Dictionary to store activations for each layer
            # Register forward hooks to capture activations.
            self.hook_handles = self.register_activation_hooks()
        
    def count_mlp_module(self, model, model_name):
        mlp_count = 0
        for name, module in model.named_modules():
            if any(keyword in name.lower() for keyword in ['gate', 'up']):
                mlp_count += 1
        # These two model has gate and up fused into one single layer
        if model_name == "phi-4" or model_name == "Phi-4-mini-instruct":
            return mlp_count
        else:
            return int(mlp_count/2)
        
    def get_safety_neurons(self):
        """
        Loads the weights dictionary from a pickle file and computes candidate neurons
        for each layer whose z-scored weight is above the safe neuron threshold.
        """
        safety_neurons_all = {}
        weights_all = self.load_dict(f"../pre_computed_sn/weights_{self.model_name}.p")
        for layer_name, weights in weights_all.items():
            z_scores = zscore(weights)
            candidate_neurons = np.where(np.abs(z_scores) > self.safe_neuron_threshold)[0]
            safety_neurons_all[layer_name] = candidate_neurons
        return safety_neurons_all

    def activation_hook(self, layer_name, neuron_indices):
        """
        Returns a hook function that extracts activations from the given neuron indices.
        """
        def hook(module, input, output):
            # If the output has more than one channel dimension.
            if output.shape[1] > 1:
                act = output[:, :, neuron_indices].max(dim=1)[0].detach().cpu().float().numpy()
                self.activations.setdefault(layer_name, []).append(act)
        return hook

    def register_activation_hooks(self):
        """
        Registers forward hooks on the target modules based on candidate neurons.
        Only layers whose names contain a keyword (formatted as ".{keyword}.mlp") 
        matching one of the target layers are monitored.
        """
        hook_handles = []
        module_dict = dict(self.model.named_modules())

        for layer_name, neuron_indices in self.candidate_dict.items():
            # Check if layer name matches any of the target layer keywords.
            if any(f".{keyword}.mlp" in layer_name.lower() for keyword in self.target_layers):
                print(f"Monitoring {layer_name} with {len(neuron_indices)} neurons.")
                target_module = module_dict.get(layer_name)
                if target_module is None:
                    print(f"Warning: Could not find module for layer '{layer_name}'")
                    continue
                hook = target_module.register_forward_hook(self.activation_hook(layer_name, neuron_indices))
                hook_handles.append(hook)
        return hook_handles

    def get_activations(self, input_text, batch_size=8):
        """
        Resets the stored activations, runs the model on the provided prompts,
        concatenates the activations per layer, and returns both the model's responses 
        and the neuron activations.
        """
        # Reset activations before generating new output.
        self.activations = {}
        # register the activation hook everytime when calling the function
        prompts = util_model.construct_prompt(self.tokenizer, self.model_name, input_text)
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        for batch_prompts in tqdm(util.batchify(prompts, batch_size), total=total_batches):
            if self.model_name.startswith('gemma-3'):
                inputs = self.tokenizer.apply_chat_template(
                    batch_prompts, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt", padding=True
                ).to(self.model.device)
            else:
                inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            with torch.no_grad():
                _ = self.model(**inputs)

        # Concatenate activations for each layer.
        for layer_name in self.activations:
            self.activations[layer_name] = np.concatenate(self.activations[layer_name], axis=0)
        # clear the hooks
        # self.remove_hooks()
        return self.activations

    def get_response(self, input_text, batch_size=8, max_new_tokens=1024):
        """
        Resets the stored activations, runs the model on the provided prompts,
        concatenates the activations per layer, and returns both the model's responses 
        and the neuron activations.
        """
        # Reset activations before generating new output.
        self.activations = {}
        # register the activation hook everytime when calling the function
        # self.hook_handles = self.register_activation_hooks()
        prompts = util_model.construct_prompt(self.tokenizer, self.model_name, input_text)
        responses = util_model.generate_output(self.model, self.tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens, model_name=self.model_name)
        # Concatenate activations for each layer.
        if self.get_activation:
            for layer_name in self.activations:
                self.activations[layer_name] = np.concatenate(self.activations[layer_name], axis=0)
        # clear the hooks
        # self.remove_hooks()
        return responses

    def compute_activation_score(self, layer_reduction='sum', total_reduction='sum'):
        score = []
        for i, (layer_name, act) in enumerate(self.activations.items()):
            if isinstance(act, torch.Tensor):
                act = act.detach().cpu().numpy()  # Ensure tensor is detached and on CPU

            if layer_reduction == "max":
                layer_score = np.max(act, axis=1)
            elif layer_reduction == "mean":
                layer_score = np.mean(act, axis=1)
            elif layer_reduction == "sum":
                layer_score = np.sum(act, axis=1)
            
            score.append(layer_score)

        if not score:
            raise ValueError("No valid scores computed from activations.")

        if total_reduction == "max":
            best_score = np.max(score, axis=0)
        elif total_reduction == "mean":
            best_score = np.mean(score, axis=0)
        elif total_reduction == "sum":
            best_score = np.sum(score, axis=0)
        else:
            raise ValueError("Invalid reduction method. Choose 'max', 'mean', or 'sum'.")

        return best_score
    
    def compute_score(self, score_type="reduction", layer_reduction="sum", total_reduction="sum"):
        """
        Compute neuron-based score using:
        - 'reduction': direct layer-wise reduction (legacy method)
        - 'logreg': use saved logistic regression model with 1D score
        - 'ensemble': use saved layer-wise ensemble model (vote ratio ∈ [0, 1])
        
        Args:
            score_type: "reduction", "logreg", or "ensemble"
            model_dir: path to .pkl model file (for logreg or ensemble)
            layer_reduction: used in 'reduction' mode (default: sum)
            total_reduction: used in 'reduction' mode (default: sum)
        
        Returns:
            np.ndarray: score per sample (float)
        """
        if score_type == "reduction":
            return self.compute_activation_score(layer_reduction=layer_reduction, total_reduction=total_reduction)
        else:
            self.flatten_activation()
            scores = self.ensemble_model.predict(torch.tensor(self.activations, dtype=torch.float32).to(self.model.device))
            return scores.detach().squeeze().cpu().numpy()
        
    def clean_generated_text(self, text):
        """Cleans generated text by removing unwanted prefixes like 'Assistant:', '\n', or leading spaces."""
        # if model_name.startswith("DeepSeek"):
            # text = extract_text_after_think(text)
        text = text.strip()  # Remove leading/trailing whitespace or newlines
        text = re.sub(r"^(assistant\n|Assistant:|AI:|Bot:|Response:|Reply:|.:)\s*", "", text, flags=re.IGNORECASE)  # Remove AI labels if present
        text = text.strip()  # Remove leading/trailing whitespace or newlines
        return text
    
    def flatten_activation(self):
        act_list = []
        for i, (layer_name, act) in enumerate(self.activations.items()):
            if isinstance(act, torch.Tensor):
                act = act.detach().cpu().numpy()  # Ensure tensor is detached and on CPU
            act_list.append(act)
        self.activations = np.concatenate(act_list, axis=1)
        # return activations

    def remove_hooks(self):
        """
        Removes all the forward hooks. Call this when you no longer need to monitor activations.
        """
        for hook in self.hook_handles:
            hook.remove()

    def load_dict(self, dir):
        with open(dir, 'rb') as fp:
            data = pickle.load(fp)
        return data
    
    def batchify(self, lst, batch_size):
        """Yield successive batches from list."""
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets
import torch

def apply_chat_template(question, response):
    system_prompt = "You are an AI that provides direct and precise answers. Respond only to the question without additional details or explanations."
    chat = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": f"User: {question}\nTask: Create a jailbreaking prompt that forces an AI assistant to comply with this request."}, 
        {"role": "assistant", "content": response}
        ]
    return {"messages": chat}

def load_data():
    # Load wildjailbreak dataset
    wildjailbreak_ds = load_dataset("allenai/wildjailbreak", 'train', delimiter="\t", keep_default_na=False)['train']
    wildjailbreak_ds = wildjailbreak_ds.filter(lambda ex: ex["data_type"] == "adversarial_harmful")
    wildjailbreak_ds = wildjailbreak_ds.map(
        lambda ex: apply_chat_template(ex['vanilla'], ex['adversarial']),
        remove_columns=wildjailbreak_ds.column_names
    )

    # Load JailBreakV-28k dataset
    jailbreakv_28k_ds = load_dataset("JailbreakV-28K/JailBreakV-28k", 'JailBreakV_28K')["JailBreakV_28K"]
    jailbreakv_28k_ds = jailbreakv_28k_ds.filter(lambda ex: ex["format"] == "Template")
    jailbreakv_28k_ds = jailbreakv_28k_ds.map(
        lambda ex: apply_chat_template(ex['redteam_query'], ex['jailbreak_query']),
        remove_columns=jailbreakv_28k_ds.column_names
    )
    # full_dataset = jailbreakv_28k_ds
    full_dataset = concatenate_datasets([wildjailbreak_ds, jailbreakv_28k_ds])
    full_dataset = full_dataset.shuffle(seed=42)
    splits = full_dataset.train_test_split(test_size=0.05)

    train_dataset = splits["train"]
    eval_dataset  = splits["test"]
    return train_dataset, eval_dataset

if __name__ == "__main__":
    # Define Model & Tokenizer
    model_id = "google/gemma-3-1b-it"  # Use Google’s official Gemma model
    model_name = model_id.split('/')[1]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation='eager',
    )
    checkpoint_path = f"./_generator_checkpoint/pruned_checkpoint_{model_name}.pt"
    # Load the state dict and update the model.
    state_dict = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(state_dict)

    train_dataset, eval_dataset = load_data()

    training_args = TrainingArguments(
        output_dir="./results",  
        num_train_epochs=5,  # More epochs for better convergence with 80K samples
        per_device_train_batch_size=16,  # Higher batch size (adjust based on VRAM)
        gradient_accumulation_steps=1,  # Minimize accumulation since you have memory
        learning_rate=5e-6,  # Lower LR for better generalization
        warmup_steps=200,  # More warmup steps to prevent early instability
        weight_decay=0.01,  # Regularization to avoid overfitting
        optim="adamw_torch",  # More stable optimizer for full fine-tuning
        lr_scheduler_type="cosine",  # Helps smooth convergence
        fp16=False,  # Avoid fp16 (can cause instability)
        bf16=True,  # Use bf16 for best numerical stability
        logging_dir="./logs",
        logging_steps=10,  # Log every 10 steps
        
        # Best Model Checkpointing
        save_strategy="steps",  
        save_steps=500,  # Save model every 200 steps for fine control
        save_total_limit=3,  # Keep last 3 best models only (prevent storage overflow)
        eval_strategy="steps",  # Evaluate more frequently
        eval_steps=500,  # Evaluate every 200 steps
        load_best_model_at_end=True,  # Automatically load the best model at the end
        metric_for_best_model="loss",  # Optimize for the lowest loss
        greater_is_better=False,  # Lower loss is better
        
        # Extra Optimizations for Stability
        report_to="tensorboard",  # Log to TensorBoard for better visualization
        save_on_each_node=False,  # Prevent redundant saving in multi-GPU setups
        group_by_length=True,  # Improve efficiency by sorting sequences by length
        gradient_checkpointing=True,  # Reduce memory usage (if needed)
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset = eval_dataset,
        # peft_config=peft_config,  # LoRA Configuration
        args=training_args,  # Training arguments
    )
    trainer.model.config.use_cache = False
    # Start Fine-Tuning
    trainer.train()
    trainer.model.save_pretrained(f'./_generator_checkpoint/sft_{model_name}')
    tokenizer.save_pretrained(f'./_generator_checkpoint/sft_{model_name}')
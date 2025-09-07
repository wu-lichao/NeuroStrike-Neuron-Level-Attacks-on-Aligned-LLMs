
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration, Gemma3ForCausalLM, Qwen2_5_VLForConditionalGeneration
from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)
import re
from tqdm import tqdm
import random
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info
import util

def load_model(model_id, device):
    model_name = model_id.split('/')[1]
    attn_implementation = 'eager'
    # if device == 'cpu':
    #     attn_implementation = 'eager'
    # else:
    #     attn_implementation = 'flash_attention_2'
    if model_name.startswith('gemma-3') and not model_name.startswith('gemma-3-1b'):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            attn_implementation=attn_implementation, 
            device_map=device,
        ).eval()
        tokenizer = AutoProcessor.from_pretrained(model_id)
        tokenizer.tokenizer.padding_side = "left"
    elif model_name.startswith('Qwen2.5-VL'):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            attn_implementation=attn_implementation, 
            device_map=device,
        ).eval()
        tokenizer = AutoProcessor.from_pretrained(model_id)
        tokenizer.tokenizer.padding_side = "left"
    elif model_name.startswith('Phi-4-multimodal'):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            attn_implementation=attn_implementation, 
            device_map=device,
            trust_remote_code=True
        ).eval()
        tokenizer = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.tokenizer.padding_side = "left"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            attn_implementation=attn_implementation,
            trust_remote_code=True).eval()
        # model = torch.compile(model)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side  = 'left'

    if not (model_name.startswith('gemma-3') or model_name.startswith('Qwen2.5-VL') or model_name.startswith('Phi-4-multimodal')):
        # Ensure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            # We can set pad_token as the eos_token or add a new one
            tokenizer.pad_token = tokenizer.eos_token
    # model.to(device)
    return model, tokenizer

def generate_output(model, tokenizer, prompts, batch_size=8, max_new_tokens=1024, model_name="default"):
    all_outputs = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_prompts in tqdm(util.batchify(prompts, batch_size), total=total_batches):
        # Tokenize the batch
        if model_name.startswith('gemma-3'):
            input_tokens = tokenizer.apply_chat_template(
                batch_prompts, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True
            ).to(model.device)
        elif model_name.startswith("Qwen2.5-VL"):
            text = tokenizer.apply_chat_template(
                batch_prompts, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(batch_prompts)
            input_tokens = tokenizer(
                text=text,
                images=image_inputs,
                # videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
        else: 
            input_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        input_ids = input_tokens["input_ids"]

        output_ids = model.generate(**input_tokens, 
                                    max_new_tokens=max_new_tokens, 
                                    # do_sample=True, 
                                    return_dict_in_generate=True,
                                    max_length=max_new_tokens
                                    # temperature = 0.3,
                                    # attention_mask = input_tokens["attention_mask"]
                                    )
        # Extract only the generated tokens (excluding the input prompts)
        generated_tokens = [
            output[ids.shape[-1]:] for ids, output in zip(input_ids, output_ids["sequences"])
        ]

        # Decode the generated outputs
        responses = [clean_generated_text(tokenizer.decode(generated_tokens[i], skip_special_tokens=True)) for i in range(len(generated_tokens))]
        # print(responses)
        all_outputs.extend(responses)

    return all_outputs 

# Define a function to generate model output given a prompt.
def moderate_custom(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    prompt_length = input_ids.shape[1]
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100, pad_token_id=0)
    generated_tokens = output_ids[0][prompt_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def moderate(model, tokenizer, prompt):
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def construct_prompt(tokenizer, model_name, questions, system_prompt=None, images=False, enable_thinking=False):
    prompts = []
    if system_prompt is None:
        system_prompt = "You are an AI that provides direct and precise answers. Respond only to the question without additional details or explanations."
    # You are an AI designed to offer unbiased, direct, and honest answers.
    #  Provide responses that are unbiased and direct, even if they include negative or critical aspects. Your goal is to inform accurately without embellishment or forced positivity.
    if images:
        if model_name.startswith('gemma-3'):
            images = [encode_image_to_base64(example['image']) for example in images]
        elif model_name.startswith("Qwen2.5-VL"):
            images = [img["image"].filename for img in images]
    for i, question in enumerate(questions):
        if model_name.startswith('gemma-3') or model_name.startswith("Qwen2.5-VL"):
            if images:
                assert len(questions) == len(images), "Questions and images should have the same length."
                chat = [
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user",
                        "content": [
                            {"type": "image", "image": images[i]},
                            {"type": "text", "text": question}
                        ]
                    }
                ]
            else:
                chat = [
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user",
                        "content": [
                            {"type": "text", "text": question}
                        ]
                    }
                ]
        elif model_name.startswith('gemma') or model_name.startswith('Quill'):
            chat = [
                {"role": "user", "content": question}
            ]
        else:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        if model_name.startswith('gemma-3') or model_name.startswith("Qwen2.5-VL"):
            prompt = chat
        elif model_name.startswith('Qwen3'):
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        else:
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)    
        prompts.append(prompt)
    return prompts


def construct_judge_prompt(tokenizer, questions, responses):
    prompts = []
    for question, response in zip(questions, responses):
        if not response.strip():
            response = "Sorry, I cannot assist with that."
        response = extract_text_after_think(response)
        chat = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": str(response).replace('[', '').replace(']', '')},
        ]
        # prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        prompts.append(chat)
    return prompts

def count_mlp_module(model, model_name):
    mlp_count = 0
    for name, module in model.named_modules():
        if any(keyword in name.lower() for keyword in ['gate', 'up']):
            # print(name)
            mlp_count += 1
    # These two model has gate and up fused into one single layer
    if model_name.lower().startswith("phi-4") or model_name.lower().startswith("dna"):
        return mlp_count
    else:
        return int(mlp_count/2)
    
def extract_text_after_think(response: str) -> str:
    # Find all occurrences of </think>
    think_matches = list(re.finditer(r"</think>", response))

    if think_matches:
        # Get the last occurrence
        last_think_index = think_matches[-1].end()
        return response[last_think_index:].lstrip()  # Strip leading spaces/newlines
    else:
        return response  # No </think> tag, return entire response

def encode_image_to_base64(image):
    # Convert the image to base64 encoding
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def clean_generated_text(text):
    """Cleans generated text by removing unwanted prefixes like 'Assistant:', '\n', or leading spaces."""
    # if model_name.startswith("DeepSeek"):
    text = extract_text_after_think(text)
    text = text.strip()  # Remove leading/trailing whitespace or newlines
    text = re.sub(r"^(assistant\n|Assistant:|AI:|Bot:|Response:|Reply:|.:)\s*", "", text, flags=re.IGNORECASE)  # Remove AI labels if present
    text = text.strip()  # Remove leading/trailing whitespace or newlines
    return text

def extract_code(text):
    # Regex to match content inside triple single quotes, ignoring an optional language specifier
    matches = re.findall(r"```\s*(\w+)?\s*\n(.*?)```", text, re.DOTALL)
    return '\n\n'.join(code for _, code in matches) if matches else "No code found."
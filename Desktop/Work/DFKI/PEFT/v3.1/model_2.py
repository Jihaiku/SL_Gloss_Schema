#!/usr/bin/env python
# coding: utf-8

# In[1]:


#
# Imports
#
import os
import random
import torch
import yaml

from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config_0 import load_config
from dataset_1 import load_and_prepare_dataset, load_processors
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#
# Load configuration
#
config = load_config()

rand_num = random.randint(1,200)
seed = rand_num
csv_filename = "model_results.csv"
model_name_or_path = config["model_name_or_path"]
model = config["model"]
language = config["language"]
language_abbr = config["language_abbr"]
task = config["task"]
dataset_name = config["dataset_name"]
size = config["size"]
user_name = config["user_name"]
peft_type = config["peft_type"]
output_dir = f"{model_name_or_path}-{language_abbr}-{size}-{seed}"

#
# Load base Whisper model with 8-bit quantization
#
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

def load_quantized_whisper_model():
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    return model

#
# Apply PEFT LoRA configuration
#
def apply_lora(model):
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config["lora_dropout"],
        bias=config["bias"]
    )

    model = get_peft_model(model, lora_config)
    return model

#
# Main execution (for standalone testing)
#
if __name__ == "__main__":
    print("\n===== Model Setup: Starting =====\n")
    print("Loading base Whisper model with 8-bit quantization...")
    model = load_quantized_whisper_model()
    print("Model loaded successfully.\n")

    print("Applying LoRA configuration...")
    model = apply_lora(model)
    print("LoRA applied successfully.\n")
    print("===== LoRA Configuration =====")
    print(f"LoRA Rank (r): {config['r']}")
    print(f"LoRA Alpha: {config['lora_alpha']}")
    print(f"LoRA Dropout: {config['lora_dropout']}")
    print(f"LoRA Target Modules: ['q_proj', 'v_proj']")
    print(f"LoRA Bias: {config['bias']} \n")

    print("Trainable parameter summary:")
    model.print_trainable_parameters()
    print("\n===== Model Setup: Complete =====\n")  


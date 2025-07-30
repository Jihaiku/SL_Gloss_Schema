#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# #
# # Imports
# #
# import torch
# from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, BitsAndBytesConfig
# from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig


# #
# # Prep Whisper
# #
# def load_processor(model_name_or_path, language, task):
#     feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
#     tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
#     processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
#     return feature_extractor, tokenizer, processor


# #
# # Load Checkpoints
# #
# def load_quantized_model(model_name_or_path):
#     bnb_config = BitsAndBytesConfig(load_in_8bit=True)
#     model = WhisperForConditionalGeneration.from_pretrained(
#         model_name_or_path,
#         quantization_config=bnb_config,
#         device_map="auto"
#     )
#     model.config.forced_decoder_ids = None
#     model.config.suppress_tokens = []
#     return model


# def apply_lora(model, lora_config):
#     model = prepare_model_for_kbit_training(model)

#     config = LoraConfig(
#         r=lora_config["r"],
#         lora_alpha=lora_config["lora_alpha"],
#         target_modules=["q_proj", "v_proj"],
#         lora_dropout=lora_config["lora_dropout"],
#         bias=lora_config.get("bias", "none"),
#     )

#     model = get_peft_model(model, config)
#     return model


# In[1]:


import os
import random
import yaml
import torch
from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

#
# Load configuration
#
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
rand_num = random.randint(1,200)
seed = rand_num
csv_filename = "model_results.csv"
filepath = config['json_path']                     # Use with JSON files
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
    model = load_quantized_whisper_model()
    model = apply_lora(model)
    model.print_trainable_parameters()


# In[ ]:





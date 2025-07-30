#!/usr/bin/env python
# coding: utf-8

# In[1]:


#
# Imports
#
import os
import torch
import gc
import csv
import evaluate
import yaml
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


from config_0 import load_config
from dataset_1 import load_and_prepare_dataset, get_prepare_dataset_fn, load_processors
from model_2 import load_quantized_whisper_model, apply_lora
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#
# Load Configs
#
config = load_config()

model_name_or_path = config["model_name_or_path"]
model = config["model"]
language = config["language"]
language_abbr = config["language_abbr"]
task = config["task"]
dataset_name = config["dataset_name"]
size = config["size"]
user_name = config["user_name"]
peft_type = config["peft_type"]
# lora_config = config.get("lora_config")  # or however it's defined in your code
# model_config = config.get("model_config")  # same


csv_filename = "model_results.csv"
rand_num = random.randint(1,200)
seed = rand_num
output_dir = f"{model_name_or_path}-{language_abbr}-{size}-{seed}"

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def train_model(
    model, processor, train_dataset, eval_dataset, output_dir,
    seed, csv_filename="model_results.csv"):

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        warmup_steps=500,
        num_train_epochs=3,
        eval_strategy="epoch",
        fp16=torch.cuda.is_available(),
        per_device_eval_batch_size=8,
        generation_max_length=128,
        logging_steps=25,
        remove_unused_columns=False,
        label_names=["labels"],
        save_total_limit=3,
    )

    # Trainer
    model.config.use_cache = False
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor
        # processing_class=processor.tokenizer
    )

    # Train
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    trainer.train()

    return training_args, trainer



if __name__ == "__main__":
    print("Running training pipeline for Whisper model with LoRA adaptation...")

    # Load model
    print("\nLoading model...")
    base_model = load_quantized_whisper_model()
    model = apply_lora(base_model)
    print("Model loaded and LoRA applied.\n")

    # Load and prepare dataset
    print("Loading and preparing dataset...")
    dataset = load_and_prepare_dataset()

    # Optional: print sampling rate
    sample_audio = dataset["train"][0]["audio"]
    print(f"Sampling rate: {sample_audio['sampling_rate']} Hz")

    # Preprocess (no multiprocessing on Windows)
    # Load feature extractor, tokenizer, processor
    feature_extractor, tokenizer, processor = load_processors(model_name_or_path, language, task)
    
    # Prepare dataset with correct scope
    prepare_dataset = get_prepare_dataset_fn(feature_extractor, tokenizer)
    dataset = dataset.map(prepare_dataset, remove_columns=["audio", "sentence"])

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}\n")

   # Train
    training_args, trainer  = train_model(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        seed=seed,
        csv_filename=csv_filename
    )

    print("\n=== Training Arguments ===")
    print(f"output_dir: {training_args.output_dir}")
    print(f"per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    print(f"learning_rate: {training_args.learning_rate}")
    print(f"warmup_steps: {training_args.warmup_steps}")
    print(f"num_train_epochs: {training_args.num_train_epochs}")
    print(f"eval_strategy: {training_args.evaluation_strategy}")
    print(f"per_device_eval_batch_size: {training_args.per_device_eval_batch_size}")
    print(f"generation_max_length: {training_args.generation_max_length}")
    print(f"logging_steps: {training_args.logging_steps}")
    print(f"save_total_limit: {training_args.save_total_limit}")

    
    print("\nTraining complete.")



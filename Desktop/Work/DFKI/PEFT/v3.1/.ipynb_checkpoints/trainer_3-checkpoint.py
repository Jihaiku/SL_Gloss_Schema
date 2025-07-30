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

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
rand_num = random.randint(1,200)
seed = rand_num
csv_filename = "model_results.csv"
filepath = config['json_path']                  # Use with JSON files
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


def train_model(model, processor, train_dataset, eval_dataset, output_dir,
    seed, lora_config, model_config, tokenizer, csv_filename="model_results.csv"):
    
    # Data Collator
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
        processing_class=processor.feature_extractor,
    )

    # Train
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    trainer.train()


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import evaluate
import gc
import jiwer
import numpy as np
import os
import random
import time
import torch
import warnings
import yaml

from datasets import load_metric
from tabulate import tabulate
from transformers import Seq2SeqTrainer
from config_0 import load_config
from dataset_1 import load_and_prepare_dataset, get_prepare_dataset_fn, load_processors
from model_2 import load_quantized_whisper_model, apply_lora
from trainer_3 import train_model, DataCollatorSpeechSeq2SeqWithPadding


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

csv_filename = "model_results.csv"
rand_num = random.randint(1,200)
seed = rand_num
output_dir = f"{model_name_or_path}-{language_abbr}-{size}-{seed}"

import torch
import numpy as np
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate

def evaluate_model_wer(model, tokenizer, test_dataset, data_collator, batch_size=8):
    eval_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)
    model.eval()
    metric = evaluate.load("wer")
    
    total_reference_tokens = 0
    
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", disable=True)):
        with torch.amp.autocast("cuda"), torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
    
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            total_reference_tokens += sum(len(label.split()) for label in decoded_labels)
    
    wer = 100 * metric.compute()
    normalized_wer = (wer / total_reference_tokens) * 100 if total_reference_tokens > 0 else None
    gc.collect()
    
    return wer, normalized_wer

def log_results_to_csv(wer, normalized_wer, dataset, seed, config_path="config.yaml", csv_filename="model_results.csv"):
    config = load_config()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.get("cuda_visible_devices", "0")

    wer = round(float(wer), 2)
    normalized_wer = round(float(normalized_wer), 2)

    lora_keys = ["r", "lora_alpha", "lora_dropout", "bias"]
    model_keys = ["model_name_or_path", "model", "language", "language_abbr", "task"]

    csv_headers = ["Seed","WER", "Norm_WER", "Training_Size", "Test_Size",] + lora_keys + model_keys

    csv_data = [seed, wer, normalized_wer, len(dataset["train"]), len(dataset["test"]), ] + [config.get(key, "NA") for key in lora_keys] + [config[key] for key in model_keys]

    write_headers = not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0

    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
    
        if write_headers:
            writer.writerow(csv_headers)

        writer.writerow(csv_data)

    # Display the results using tabulate
    if not os.path.exists(csv_filename):
        print(f"Error: {csv_filename} not found.")
    else:
        with open(csv_filename, mode="r", newline="") as file:
            reader = csv.reader(file)
            data = list(reader)

        if len(data) < 1:
            print("CSV file exists but has no data to display.")
        else:
            headers = data[0]
            rows = data[1:]
            print(tabulate(rows, headers=headers, tablefmt="grid"))



if __name__ == "__main__":
    base_model = load_quantized_whisper_model()
    model = apply_lora(base_model)
    dataset = load_and_prepare_dataset()
    feature_extractor, tokenizer, processor = load_processors(model_name_or_path, language, task)
    prepare_dataset = get_prepare_dataset_fn(feature_extractor, tokenizer)
    dataset = dataset.map(prepare_dataset, remove_columns=["audio", "sentence"])

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    training_args, trainer  = train_model(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        seed=seed,
        csv_filename=csv_filename
    )


    # Run evaluation with WER on the trained model
    wer, normalized_wer = evaluate_model_wer(model, tokenizer, dataset["test"], data_collator)
    print(f"WER: {wer:.2f}%, Normalized WER: {normalized_wer:.4f}%")

    # Logging
    log_results_to_csv(wer, normalized_wer, dataset, seed, csv_filename="eval_test.csv")


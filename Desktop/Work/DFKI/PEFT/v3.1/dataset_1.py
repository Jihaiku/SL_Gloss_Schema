#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import random
import torch
import yaml

from dataclasses import dataclass
from typing import Any, List, Dict, Union
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

from config_0 import load_config
# Load config
config = load_config()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name_or_path = config["model_name_or_path"]
language = config["language"]
language_abbr = config["language_abbr"]
task = config["task"]
dataset_name = config["dataset_name"]
size = config["size"]
seed = random.randint(1, 200)

def load_and_prepare_dataset():
    # Here is where I gotta change things later
    dataset = load_dataset(dataset_name, language_abbr) # <- This is where you should change it
    dataset = dataset["train"].train_test_split(train_size=size, seed=seed) # <- change size to 1.0
    dataset["test"] = load_dataset(dataset_name, language_abbr, split="test").train_test_split(train_size=size, seed=seed)["train"]

    # Check required columns
    required_columns = ["audio", "sentence"]
    missing_columns = {}
    for split in ["train", "test"]:
        missing = [col for col in required_columns if col not in dataset[split].column_names]
        if missing:
            missing_columns[split] = missing

    if missing_columns:
        missing_str = ", ".join([f"{split}: {', '.join(missing)}" for split, missing in missing_columns.items()])
        raise ValueError(
            f"Missing required columns in splits: {missing_str}. The dataset must contain 'audio' and 'sentence' columns in both 'train' and 'test'."
        )

    for split in ["train", "test"]:
        dataset[split] = dataset[split].remove_columns(
            [col for col in dataset[split].column_names if col not in required_columns]
        )

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset




def load_processors(model_name_or_path, language, task):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
    return feature_extractor, tokenizer, processor


# def prepare_dataset(batch):
#     audio = batch["audio"]
#     batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
#     batch["labels"] = tokenizer(batch["sentence"]).input_ids
#     return batch

def get_prepare_dataset_fn(feature_extractor, tokenizer):
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
    return prepare_dataset

if __name__ == "__main__":
    print("Loading and preparing dataset...")
    dataset = load_and_prepare_dataset()

    # Print sampling rate
    sample_audio = dataset["train"][0]["audio"]
    print(f"\nSampling rate: {sample_audio['sampling_rate']} Hz")
    
    print("\nDataset info:")
    print(f"Dataset Name: {dataset_name}")
    print(f"Percentage of the dataset: {size}%")
    for split in ["train", "test"]:
        print(f"Split: {split}")
        print(f"  Number of samples: {len(dataset[split])}")
        # print(f"  Column names: {dataset[split].column_names}")
        # print(f"  First sample:\n{dataset[split][0]}\n")


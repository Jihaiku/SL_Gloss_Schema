#
# Imports
#
import random
import yaml
from dataclasses import dataclass
from typing import Any, List, Dict, Union

import torch
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

# Load config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name_or_path = config["model_name_or_path"]
language = config["language"]
language_abbr = config["language_abbr"]
task = config["task"]
dataset_name = config["dataset_name"]
size = config["size"]
seed = random.randint(1, 200)



def load_and_prepare_dataset():
    dataset = load_dataset(dataset_name, language_abbr)
    dataset = dataset["train"].train_test_split(train_size=size, seed=seed)
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

# Prepare Whisper processor objects
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove bos token if present at start of all labels
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

if __name__ == "__main__":
    print("Loading and preparing dataset...")
    dataset = load_and_prepare_dataset()
    
    print("\nDataset info:")
    print(f"Dataset Name: {dataset_name}")
    print(f"Percentage of the dataset: {size}%")
    for split in ["train", "test"]:
        print(f"Split: {split}")
        print(f"  Number of samples: {len(dataset[split])}")
        # print(f"  Column names: {dataset[split].column_names}")
        # print(f"  First sample:\n{dataset[split][0]}\n")


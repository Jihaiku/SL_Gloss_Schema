#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import random
import gc
import warnings

from datasets import Audio
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
from tqdm import tqdm

from config_0 import load_config
from dataset_1 import load_and_prepare_dataset, get_prepare_dataset_fn, load_processors
from model_2 import load_quantized_whisper_model, apply_lora
from trainer_3 import train_model, DataCollatorSpeechSeq2SeqWithPadding
from eval_4 import evaluate_model_wer, log_results_to_csv


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def main(verbose=True):
    # Step 0: Load configuration
    print("0. Loading Config")
    config = load_config("config.yaml")
    model_name_or_path = config["model_name_or_path"]
    language = config["language"]
    language_abbr = config["language_abbr"]
    task = config["task"]
    dataset_name = config["dataset_name"]
    size = config["size"]

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HOME"] = os.getcwd()
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"

    seed = random.randint(1, 200)
    csv_filename = "model_results.csv"
    output_dir = f"{config['model_name_or_path']}-{config['language_abbr']}-{config['size']}-{seed}"

    if verbose == True:
        try:
            cfg = load_config()
            assert isinstance(cfg, dict), "Config is not a dictionary."
            print("\n===== Config Info =====")
            print("Config contents:\n")
            max_key_len = max(len(key) for key in cfg.keys())
            for key, value in cfg.items():
                print(f"{key.ljust(max_key_len)} : {value}")
        except Exception as e:
            print(f"Test failed: {e}")
        print("===== Config Info ===== ")
    
    
    print("\n0. Configuration loaded.")


    # Step 1: Load and prepare dataset
    print("\n1. Loading Dataset")
    dataset = load_and_prepare_dataset()

    if verbose == True:
        print("\n===== Datset Info =====")
        dataset = load_and_prepare_dataset()
    
        # Print sampling rate
        sample_audio = dataset["train"][0]["audio"]
        print(f"Sampling rate: {sample_audio['sampling_rate']} Hz")
        print(f"Dataset Name: {dataset_name}")
        print(f"Percentage of the dataset: {size}%")
        for split in ["train", "test"]:
            print(f"Split: {split}")
            print(f"  Number of samples: {len(dataset[split])}")
        print("===== Dataset Info =====")
        
    print("\n1. Dataset loaded and prepared.")

    
    # Step 2: Load quantized Whisper model & apply LoRA
    print("\n2. Loading Model")
    base_model = load_quantized_whisper_model()
    model = apply_lora(base_model)

    if verbose == True:
        print("\n===== Model Info =====")
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
        print("===== Model Info =====\n")   
    
    print("2. Model loaded and LoRA applied.")


    # Step 3: Train
    print("\nLoading model...")
    base_model = load_quantized_whisper_model()
    model = apply_lora(base_model)
    
    dataset = load_and_prepare_dataset()
    feature_extractor, tokenizer, processor = load_processors(model_name_or_path, language, task)
    
    prepare_dataset = get_prepare_dataset_fn(feature_extractor, tokenizer)
    dataset = dataset.map(prepare_dataset, remove_columns=["audio", "sentence"])
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # print(f"\nTraining samples: {len(train_dataset)}")
    # print(f"Evaluation samples: {len(eval_dataset)}\n")

   # Train
    train_model(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        seed=seed,
        csv_filename=csv_filename
    )

    # if verbose == True:
    #     print("\n=== Training Arguments ===")
    #     print(f"output_dir: {training_args.output_dir}")
    #     print(f"per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    #     print(f"gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    #     print(f"learning_rate: {training_args.learning_rate}")
    #     print(f"warmup_steps: {training_args.warmup_steps}")
    #     print(f"num_train_epochs: {training_args.num_train_epochs}")
    #     print(f"eval_strategy: {training_args.evaluation_strategy}")
    #     print(f"per_device_eval_batch_size: {training_args.per_device_eval_batch_size}")
    #     print(f"generation_max_length: {training_args.generation_max_length}")
    #     print(f"logging_steps: {training_args.logging_steps}")
    #     print(f"save_total_limit: {training_args.save_total_limit}")
    #     print("=== Training Arguments ===")

    # print("3. Training completed.")

    # Step 4: Evaluate
    wer, normalized_wer = evaluate_model_wer(model, tokenizer, dataset["test"], data_collator)
    print(f"WER: {wer:.2f}%, Normalized WER: {normalized_wer:.4f}%")

    # Step 5: Log
    wer, normalized_wer = evaluate_model_wer(model, tokenizer, dataset["test"], data_collator)
    log_results_to_csv(wer, normalized_wer, dataset, seed)


    # gc.collect()


if __name__ == "__main__":
    main(verbose=False)


# In[ ]:





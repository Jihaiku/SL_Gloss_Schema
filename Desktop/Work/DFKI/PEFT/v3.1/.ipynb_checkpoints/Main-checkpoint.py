#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# #
# # Imports
# #

# # Normal Library Imports
# import os
# import torch
# import random
# import gc
# import warnings

# from datasets import Audio

# # Subfolder Imports
# from config_0 import load_config
# from dataset_1 import DatasetLoader, prepare_dataset
# from model_2 import load_model, prepare_peft_model
# from trainer_3 import get_data_collator, get_trainer
# # from 4_evaluate import compute_metrics
# # from 5_logging import log_results

# # tqdm().set_description("Training Progress")
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
# # logging.set_verbosity_error()

# def main():
#     #
#     # config_0 - Load configurations
#     #
#     config = load_config("config.yaml")

#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     rand_num = random.randint(1,200)
#     seed = rand_num
#     csv_filename = "model_results.csv"
#     filepath = config['json_path']
#     model_name_or_path = config["model_name_or_path"]
#     model = config["model"]
#     language = config["language"]
#     language_abbr = config["language_abbr"]
#     task = config["task"]
#     dataset_name = config["dataset_name"]
#     size = config["size"]
#     user_name = config["user_name"]
#     peft_type = config["peft_type"]
#     output_dir = f"{model_name_or_path}-{language_abbr}-{size}-{seed}"

#     cache_dir = os.getcwd()
#     os.environ['HF_HOME'] = cache_dir
    
    
#     print(f"0. Configuration and YAML file are loaded.")



    
#     #
#     # dataset_1 - Load Dataset
#     #
#     loader = DatasetLoader(size=size, seed=seed)
#     dataset = loader.load_from_huggingface(dataset_name, language_abbr)
#                                        # Change the Method within the loader based on your dataset

#     # Prepare Whisper
#     feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
#     tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
#     processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

#     # Prepare dataset
#     dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
#     dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=os.cpu_count())

#     print(f"1. Dataset is loaded, modified, and prepared")




#     # 2_model - Load model
#     bnb_config = BitsAndBytesConfig(load_in_8bit=True)
#     base_model_1 = load_quantized_whisper_model()
#     base_model_2 = apply_lora(base_model_1)
#     base_model = load_model(config["model_name_or_path"])
#     model = prepare_peft_model(base_model, config)

#     # Set up data collator
#     # data_collator = get_data_collator()    # Where to put the data collator

#     # 3_training - Training arguments
#     output_dir = f"{config['model_name_or_path']}-{config['language_abbr']}-{config['size']}-{seed}"

#     # trainer = get_trainer(
#     #     model=model,
#     #     dataset=dataset,
#     #     data_collator=data_collator,
#     #     config=config,
#     #     output_dir=output_dir
#     # )

#     # # Train
#     # trainer.train()

#     # # 4_evaluate - Evaluate (optional WER metrics)
#     # wer, norm_wer = compute_metrics(trainer, dataset["test"])

#     # # 5_logging - Log results
#     # log_results(
#     #     seed=seed,
#     #     wer=wer,
#     #     norm_wer=norm_wer,
#     #     dataset=dataset,
#     #     config=config,
#     #     output_path="model_results.csv"
#     # )

#     # # Cleanup
#     # gc.collect()

# if __name__ == "__main__":
#     main()


# In[ ]:


#
# Imports
#
import os
import torch
import random
import gc
import warnings
from datasets import Audio
from tqdm import tqdm

# Subfolder Imports
from config_0 import load_config
from dataset_1 import DatasetLoader, prepare_dataset
from model_2 import load_quantized_whisper_model, apply_lora
from trainer_3 import train_model  # assuming train_model is exposed from trainer.py

# Silence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    #
    # Step 0: Load configuration
    #
    config = load_config("config.yaml")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HOME"] = os.getcwd()
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"

    rand_num = random.randint(1, 200)
    seed = rand_num
    csv_filename = "model_results.csv"
    output_dir = f"{config['model_name_or_path']}-{config['language_abbr']}-{config['size']}-{seed}"

    print("0. Configuration loaded.")


    #
    # Step 1: Load and prepare dataset
    #
    loader = DatasetLoader(size=config["size"], seed=seed)
    dataset = loader.load_from_huggingface(config["dataset_name"], config["language_abbr"])

    # Load processor components
    feature_extractor, tokenizer, processor = load_processor(
        config["model_name_or_path"],
        config["language"],
        config["task"]
    )

    # Prepare audio dataset
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count()
    )

    print("1. Dataset loaded and prepared.")


    #
    # Step 2: Load quantized model and apply LoRA
    #
    base_model = load_quantized_whisper_model()
    model = apply_lora(base_model)

    print("2. Model loaded and LoRA applied.")


    #
    # Step 3: Train the model
    #
    train_model(
        model=model,
        processor=processor,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        output_dir=output_dir,
        seed=seed,
        lora_config=config,  # assuming LoRA settings are in config
        model_config=config,
        tokenizer=tokenizer,
        csv_filename=csv_filename
    )

    print("3. Training completed.")

    #
    # Optional: Cleanup
    #
    gc.collect()


if __name__ == "__main__":
    main()


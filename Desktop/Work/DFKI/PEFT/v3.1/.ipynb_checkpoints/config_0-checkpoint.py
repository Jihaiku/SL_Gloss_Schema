#!/usr/bin/env python
# coding: utf-8

# In[11]:


#
# Imports
#
import yaml
import os

#
# Open Configs
#
def load_config(config_path="config.yaml"):
    """
    Loads the YAML config file and returns it as a Python dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config

#
# Sanity Testing
#
if __name__ == "__main__":
    print("This code is meant to open up the config.yaml file so that \nyou may be able to run variable features found within without \nneeding to constantly change the code.\n")
    print("Running config.py as standalone script...")

    try:
        cfg = load_config()
        assert isinstance(cfg, dict), "Config is not a dictionary."
        print("Config loaded successfully.")
        print("\nConfig contents:\n")
        max_key_len = max(len(key) for key in cfg.keys())
        for key, value in cfg.items():
            print(f"{key.ljust(max_key_len)} : {value}")
    except Exception as e:
        print(f"Test failed: {e}")


"""Module for loading config file
"""
# Standard imports
import json

def get_config():
    """Function to read config dictionary
    
    Returns:
        dict: Config dictionary
    """
    config = json.loads(open("./utils/config/config.json","r").read())
    return config

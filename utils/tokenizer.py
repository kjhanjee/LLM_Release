"""Module for initializing tokenizer variable
"""
# Standard imports
from tokenizers import Tokenizer
# Config import
from utils.read_config import get_config

config = get_config()

def get_tokenizer():
    """Function to get the tokenizer variable returned
    
    Returns:
        BPE Tokenizer: BPE Tokenizer object trained on current training data
    """
    tokenizer = Tokenizer.from_file(config['tokenizer_path'])
    return tokenizer
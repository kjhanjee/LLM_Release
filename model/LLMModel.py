"""Module for creating final Pytorch Model object
"""
# Standard Imports
import torch
from torch.cuda.amp.autocast_mode import custom_fwd
# Custom layer imports
from model.LLMLayer import LLMLayer


class LLMModel(torch.nn.Module):
    """
    Module that uses LLM layer to create the LLM Pytorch Model
    """

    def __init__(self, model:LLMLayer):
        """Init method to initialize class variables
        
        Args:
            model (LLMLayer): LLM model with individual layer definitions
        
        Returns:
            None
        """
        super().__init__()
        self.model = model
        self.max_sequence_length = self.model.max_sequence_length
        
    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        """Forward function for the model to predict/train on Input tokens
        
        Args:
            x (Tensor): Input Token Tensor
            mask (Tensor): Mask tensor for Attention weights Masking
        
        Returns:
            Tensor: Predicted Tokens with context window moved right + 1
        """
        inp = x
        output = self.model(inp, mask)
        return output
    
    @custom_fwd(cast_inputs=torch.float16)
    def predict(self, x:torch.Tensor, mask:torch.Tensor, temperature:float = 1.0):
        """Forward function for the model to predict/train on Input tokens
        
        Args:
            x (Tensor): Input Token Tensor
            mask (Tensor): Mask tensor for Attention weights Masking
            temperature (float): Temperature argument to control the Randomness of the model
        
        Returns:
            Tensor: Predicted Tokens with context window moved right + 1 with Temperature applied
        """
        output = self.model(x, mask)
        if temperature <= 1.0:
            output = output * temperature
        else:
            print("INFO-- temperature not set between 0 and 1. Ignoring temperature")
        return output
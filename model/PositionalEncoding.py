""" Module for creating the Positional Encoding Layer class
"""
# Standard Imports
import math
import torch
from torch import zeros, arange, exp, sin, cos
from torch.cuda.amp.autocast_mode import custom_fwd

class PositionalEncodingLayer(torch.nn.Module):
    """Module for creating Positional Encoding layer for the model
    """
    def __init__(self, embedding_dimension:int):
        """Init function to initialize the class variables
        
        Args:
            embedding_dimension (int): Embeddings Layer Dimension for the Model
        
        Returns:
            None
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        
    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, x:torch.Tensor):
        """Forward function for the layer
        
        Args:
            x (Tensor): Tokens Embeddings Tensor for the layer input
        
        Returns:
            Tensor: Input Embeddings with Added Positional Weights
        """
        pe = zeros(x.shape[1], self.embedding_dimension) # Define Tensor of Zeros with input shape
        position = arange(0, x.shape[1]).unsqueeze(1) # Unsqueezing the input
        div_term = exp((arange(0, self.embedding_dimension, 2, dtype=torch.float32)*-(math.log(10000.0) / self.embedding_dimension))) # Calculating Division Term for the given Embeddings dimension
        pe[:, 0::2] = sin(position.float() * div_term) # Calculating Positional Encoding for even positioned tokens
        pe[:, 1::2] = cos(position.float() * div_term) # Calculating Positional Encoding for odd positioned tokens
        out = x + pe[:x.size(1), :] # Residual Addition
        return out
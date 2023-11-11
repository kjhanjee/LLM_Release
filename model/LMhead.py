"""Module for creating the LMHead Layer class
"""
# Standard imports
import torch
from torch.cuda.amp.autocast_mode import custom_fwd

class LMHeadLayer(torch.nn.Module):
    """Module for creating the LMHead Layer
    """
    def __init__(self, embedding_dimension:int, number_of_tokens:int):
        """Function to initialize the class variables
        
        Args:
            embedding_dimension (int): Dense Token Embedding Dimension for the LMHead layer as input
            number_of_tokens (int): Tokenizer Vocabulary for prediction features
        
        Returns:
            None
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_tokens = number_of_tokens
        self.linear = torch.nn.Linear(embedding_dimension, number_of_tokens)
        
    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, x:torch.Tensor):
        """Forward Function for the class to produce token probabilities
        
        Args:
            x (Tensor): Input Tensor for the model to predict token probabilities
        """
        linear_output = self.linear(x)
        return linear_output
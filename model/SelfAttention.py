"""Module for creating Self Attention layer class for the Multi Head Attention Layer
"""
# Standard Imports
import torch
from torch import matmul, ones, bmm
from torch.cuda.amp.autocast_mode import custom_fwd
import numpy as np

    
class SelfAttention(torch.nn.Module):
    """Module for creating Self Attention layer for the Multi Head Attention Layer
    """
    def __init__(self, embedding_dimension:int, head_dimension:int, training:bool):
        """Init function to intialize the class variables
        
        Args:
            embedding_dimension (int): Embedding Dimension being used by the Model
            head_dimension (int): Head Dimension calculated based on number of heads from the Multi Head Attention Layer
        
        Return:
            None
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.training = training
        self.k_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.q_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.v_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.softmax = torch.nn.Softmax(dim=-1)
        
    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        """Forward function for the layer
        
        Args:
            x (Tensor): Input Embeddings Tensor
            mask (Tensor): Mask Tensor for Attention Masking (values 0 or 1)
        
        Return:
            Tensor: Attention Weights calculated as per Attention is all you need paper
        """
        key = self.k_layer(x)
        query = self.q_layer(x)
        value = self.v_layer(x)
        attention_weights = matmul(query, key.transpose(-2, -1)) # Creating Key . Query
        attention_weights = attention_weights / np.sqrt(self.head_dimension) # Weight Scaling
        temp_mask = ones((attention_weights.shape[0],attention_weights.shape[1],attention_weights.shape[2]), dtype = torch.float32, device='cuda') # Temporary Mask object with all 1s
        for index, item in enumerate(temp_mask):
            for index2 in range(len(item)):
                temp_mask[index][index2] = mask[index]
            temp_mask[index] = temp_mask[index].triu(0) # Sequence X Sequence Matrix with Mask values as 0 for upper diagonal
        if self.training:
            attention_weights = attention_weights.masked_fill(temp_mask == 0, -1e22) # Very Small number assignment so that further processing is not done on the Attention Weight
        attention_scores = self.softmax(attention_weights) # Weight to scores
        out = bmm(attention_scores, value) # Batch matric product of Attention scores and the input
        return out
"""Module for creating Multi Head Attention Layer class
"""
import torch
from torch import cat
from torch.cuda.amp.autocast_mode import custom_fwd
# Custom Module imports
from model.SelfAttention import SelfAttention
    
class MultiHeadAttention(torch.nn.Module):
    """Module for creating Muti Head Attention Layer
    """

    def __init__(self, embedding_dimension:int, number_of_heads:int, training: bool):
        """Init function to initialize the class variables
        
        Args:
            embeddings_dimension (int): Embeddings dimension for the Token Embeddings vector used as input
            number_of_heads (int): Number of heads to divide the Embeddings Dimension into for minute pattern detection
            training (bool): Whether the model is undergoing training or not
            
        Returns:
            None
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = embedding_dimension // number_of_heads
        self.number_of_heads = number_of_heads
        self.training = training
        self.self_attentions = torch.nn.ModuleList([SelfAttention(embedding_dimension, self.head_dimension, self.training) for _ in range(number_of_heads)])
        self.output_layer = torch.nn.Linear(number_of_heads * self.head_dimension, embedding_dimension)

    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        """Forward function for the layer
        
        Args:
            x (Tensor): Input Tokens Embeddings Tensor for the Layer as input
            mask (Tensor): Mask Tensor for Attention Weights Masking process (values 0 or 1)
        
        Returns:
            Tensor: Attention Weights with same dimensions as the embeddings
        """
        self_attention_outputs = [self_attention(x, mask) for self_attention in self.self_attentions] # Individual Attention Head forward process 
        concatenated_self_attention_outputs = cat(self_attention_outputs, dim=2) # Concatenating the outputs from all Heads
        output = self.output_layer(concatenated_self_attention_outputs) # Linear computation between Heads * Heads Dimension X Embddings dimension. Which is essentially the same dimension over a linear computation
        return output
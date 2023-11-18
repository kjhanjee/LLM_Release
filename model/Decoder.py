""" Module designed for creating Decoder Layer Architecture
"""
# Standard imports
import torch
from torch.cuda.amp.autocast_mode import custom_fwd
# Custom NN Module classes
from model.MultiHeadAttention import MultiHeadAttention
from model.FeedForward import FeedForwardLayer

class DecoderLayer(torch.nn.Module):
    """
    Module for Decoder Layer
    
    Input dimensions: batch X sequence length X embeddings dimension
    Output dimensions: batch X sequence length X embeddings dimension
    """

    def __init__(self,embedding_dimension: int,number_of_heads:int,dropout_rate:int,decoder_dim:int,training:bool = True):
        """Init function to initialize the Decoder class
        
        Args:
            embedding_dimension (int): Dimension of the embedding vector created for each token
            number_of_heads (int): Number of heads for Multi head attention
            dropout_rate (float): The Dropout rate in float
            decoder_dim (int): Dimension output from each parallel decoder layer
            training (bool): Whether the layer is being trained or not
        
        Returns:
            None
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.dropout_rate = dropout_rate
        self.training = training
        self.decoder_dim = decoder_dim
        self.multi_headed_self_attention = MultiHeadAttention(embedding_dimension, number_of_heads, self.training)
        self.residual_reduction = torch.nn.Linear(embedding_dimension,self.decoder_dim)
        self.feed_forward_reduction = torch.nn.Linear(embedding_dimension,self.decoder_dim)
        self.feed_forward = FeedForwardLayer(self.embedding_dimension, 4*self.embedding_dimension) # Feed Forward is autocalculated at 4*Embedding Dimension
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_normalization = torch.nn.LayerNorm(self.embedding_dimension)

    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, x: torch.Tensor, mask:torch.Tensor):
        """Forward method for the layer
        
        Args:
            x (Tensor): Input Tensor in the dimension of Batch Size X Sequence Length X Embeddings Dimension
            mask (Tensor): mask Tensor object (possible values 0 ior 1) for attention weight masking
            
        Returns:
            Tensor: Output from the decoder layer of dimension Batch Size X Sequence Length X Embeddings Dimension/number of layers
        """
        attention_output = self.multi_headed_self_attention(x, mask)
        residual_output = x + attention_output
        normalized_residual_output = self.layer_normalization(residual_output)
        residual_output = self.residual_reduction(normalized_residual_output) # Reducing attention outputs to Embeddings layer/number of layers in dim
        feed_forward_output = self.feed_forward(normalized_residual_output)
        feed_forward_output = self.feed_forward_reduction(feed_forward_output)
        if self.training:
            feed_forward_output = self.dropout(feed_forward_output)
        out = residual_output + feed_forward_output
        return out
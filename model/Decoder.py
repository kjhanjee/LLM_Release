""" Module designed for creating Decoder Layer Architecture
"""
# Standard imports
import torch
from torch.cuda.amp.autocast_mode import custom_fwd
# Custom NN Module classes
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
        self.residual_reduction = torch.nn.Linear(embedding_dimension,self.decoder_dim).to(dtype=torch.float16)
        self.feed_forward_reduction = torch.nn.Linear(embedding_dimension,self.decoder_dim)
        self.feed_forward = FeedForwardLayer(self.decoder_dim, 4*self.decoder_dim).to(dtype=torch.float16) # Feed Forward is autocalculated at 4*Embedding Dimension
        self.dropout = torch.nn.Dropout(dropout_rate)

    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, x: torch.Tensor):
        """Forward method for the layer
        
        Args:
            x (Tensor): Input Tensor in the dimension of Batch Size X Sequence Length X Embeddings Dimension
            mask (Tensor): mask Tensor object (possible values 0 ior 1) for attention weight masking
            
        Returns:
            Tensor: Output from the decoder layer of dimension Batch Size X Sequence Length X Embeddings Dimension/number of layers
        """
        residual_output = self.residual_reduction(x.to(dtype=torch.float16)) # Reducing attention outputs to Embeddings layer/number of layers in dim
        feed_forward_output = self.feed_forward(residual_output.to(dtype=torch.float16))
        if self.training:
            feed_forward_output = self.dropout(feed_forward_output)
        out = residual_output + feed_forward_output
        return out.to(dtype=torch.float32)
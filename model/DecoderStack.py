"""Module for Creating Decoder Layer Stack Architecture
"""
# Standard Imports
import torch
from torch.cuda.amp.autocast_mode import custom_fwd
# Custom Module Imports
from model.FeedForward import FeedForwardLayer
from model.Decoder import DecoderLayer
    
class DecoderStackLayer(torch.nn.Module):
    """
    Module created using stacking Decoder layers. Last Layer results in 4*Embeddings Dimension 
    """

    def __init__(self,embedding_dimension: int,number_of_layers: int,number_of_heads: int,dropout_rate: float,max_sequence_length: int,stack_index: int,max_stack: int,training:bool = True):
        """Init function to initialize the Decoder Stack class
        
        Args:
            embedding_dimension (int): Dimension of the embedding vector created for each token
            number_of_layers (int): Number of Decoder layers to be stacked in this module
            number_of_heads (int): Number of heads for Multi head attention
            dropout_rate (float): The Dropout rate in float
            max_sequence_length (int): Maximum Sequence Size supported by the model as input
            stack_index (int): Index Value of the Current Stack Layer
            max_stack (int): Maximum Number of Stacked Layers in Serial input
            training (bool): Whether the layer is being trained or not
        
        Returns:
            None
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length
        self.training = training
        self.stack_index = stack_index
        self.max_stack = max_stack
        
        
        self.decoder_dim1 = int(embedding_dimension/number_of_layers)
        self.feed_forward1 = FeedForwardLayer(embedding_dimension,4*embedding_dimension)
        
        self.normalize2_1 = torch.nn.LayerNorm(embedding_dimension)
        self.decoder_dim2 = int(4*embedding_dimension/number_of_layers)
        self.feed_forward2 = FeedForwardLayer(4*embedding_dimension,16*embedding_dimension)
        self.normalize2_2 = torch.nn.LayerNorm(4*embedding_dimension)

        self.decoder_layers = torch.nn.ModuleList([DecoderLayer(embedding_dimension,number_of_heads,dropout_rate,self.decoder_dim1,training)
                                                   if self.stack_index != self.max_stack
                                                   else DecoderLayer(embedding_dimension,number_of_heads,dropout_rate,self.decoder_dim2,training)
                                                   for _ in range(number_of_layers)])
        
    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, x, mask):
        """Forward method for the layer
        
        Args:
            x (Tensor): Input Tensor in the dimension of Batch Size X Sequence Length X Embeddings Dimension
            mask (Tensor): mask Tensor object (possible values 0 ior 1) for attention weight masking
            
        Returns:
            Tensor: Output from the decoder layer of dimension Batch Size X Sequence Length X Embeddings Dimension for layers 1 to n-1. For nth layer the dimensions are Batch Size X Sequence Length X 4* Embeddings Dimension
        """
        if self.stack_index > 0:
            decoder_outputs = self.normalize2_1(x)
        decoder_outputs = [decoder(x, mask) for decoder in self.decoder_layers]
        decoder_outputs = torch.cat(decoder_outputs,dim=2)
        if self.stack_index == self.max_stack:
            decoder_outputs = self.feed_forward2(decoder_outputs)
            decoder_outputs = self.normalize2_2(decoder_outputs)
        else:
            decoder_outputs = self.feed_forward1(decoder_outputs)
        return decoder_outputs
    
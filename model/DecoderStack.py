"""Module for Creating Decoder Layer Stack Architecture
"""
# Standard Imports
import torch
from torch.cuda.amp.autocast_mode import custom_fwd
# Custom Module Imports
from model.MultiHeadAttention import MultiHeadAttention
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
        
        self.multi_headed_self_attention_1 = MultiHeadAttention(embedding_dimension, number_of_heads, self.training)
        self.layer_normalization_1 = torch.nn.LayerNorm(self.embedding_dimension)
        
        self.multi_headed_self_attention_2 = MultiHeadAttention(8*embedding_dimension, number_of_heads, self.training)
        self.layer_normalization_2 = torch.nn.LayerNorm(8*self.embedding_dimension)
        
        self.decoder_dim1 = int(embedding_dimension/number_of_layers)
        self.feed_forward1 = FeedForwardLayer(embedding_dimension,4*embedding_dimension).to(dtype=torch.float16)
        self.normalize1 = torch.nn.LayerNorm(embedding_dimension)
        
        self.normalize2_1 = torch.nn.LayerNorm(embedding_dimension)
        self.normalize2_2 = torch.nn.LayerNorm(8*embedding_dimension)
        self.decoder_dim2 = int(8*embedding_dimension/number_of_layers)
        self.feed_forward2 = FeedForwardLayer(8*embedding_dimension,32*embedding_dimension).to(dtype=torch.float16)
        self.normalize2_2 = torch.nn.LayerNorm(8*embedding_dimension)
        self.decoder_layers = []
        for _ in range(number_of_layers):
            if self.stack_index == 0:
                self.decoder_layers.append(DecoderLayer(embedding_dimension,number_of_heads,dropout_rate,self.decoder_dim1,training))
            elif self.stack_index == 1:
                self.decoder_layers.append(DecoderLayer(embedding_dimension,number_of_heads,dropout_rate,self.decoder_dim2,training))
            else:
                self.decoder_layers.append(DecoderLayer(8*embedding_dimension,number_of_heads,dropout_rate,self.decoder_dim2,training))
        self.decoder_layers = torch.nn.ModuleList(self.decoder_layers)
        
    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, x, mask):
        """Forward method for the layer
        
        Args:
            x (Tensor): Input Tensor in the dimension of Batch Size X Sequence Length X Embeddings Dimension
            mask (Tensor): mask Tensor object (possible values 0 ior 1) for attention weight masking
            
        Returns:
            Tensor: Output from the decoder layer of dimension Batch Size X Sequence Length X Embeddings Dimension for layers 1 to n-1. For nth layer the dimensions are Batch Size X Sequence Length X 4* Embeddings Dimension
        """
        if self.stack_index < 2:
            decoder_outputs = self.normalize2_1(x)
            attention_output = self.multi_headed_self_attention_1(x, mask)
            residual_output = x + attention_output
            normalized_residual_output = self.layer_normalization_1(residual_output)
        elif self.stack_index >= 2:
            decoder_outputs = self.normalize2_2(x)
            attention_output = self.multi_headed_self_attention_2(x, mask)
            residual_output = x + attention_output
            normalized_residual_output = self.layer_normalization_2(residual_output)
        decoder_outputs = [decoder(normalized_residual_output) for decoder in self.decoder_layers]
        decoder_outputs = torch.cat(decoder_outputs,dim=-1)
        if self.stack_index > 0:
            decoder_outputs = self.feed_forward2(decoder_outputs.to(dtype=torch.float16))
            decoder_outputs = self.normalize2_2(decoder_outputs.to(dtype=torch.float32))
        else:
            decoder_outputs = self.feed_forward1(decoder_outputs.to(dtype=torch.float16))
            decoder_outputs = self.normalize1(decoder_outputs.to(dtype=torch.float32))
        return decoder_outputs
    
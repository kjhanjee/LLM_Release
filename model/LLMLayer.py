"""Module for creating Complete LLM layer module
"""
# Standard Imports
import torch
from torch import cat
from torch.cuda.amp.autocast_mode import custom_fwd
# Custom Module Imports
from model.Embeddings import EmbeddingLayer
from model.PositionalEncoding import PositionalEncodingLayer
from model.DecoderStack import DecoderStackLayer
from model.LMhead import LMHeadLayer

class LLMLayer(torch.nn.Module):
    """
    Pytorch module for a language model.
    """

    def __init__(self,number_of_tokens: int,max_sequence_length: int=512,embedding_dimension: int=512,number_of_layers: int=6,number_of_heads: int=4,decoder_stacks:int = 2,feed_forward_dimension: int=None,dropout_rate: float=0.1,training:bool = True):
        """Init function to initialize class variables
        
        Args:
            number_of_tokens (int): Tokenizer Vocabulary the model is being trained on or is predicting
            max_sequence_length (int): Maximum sequence Length for the model input
            embedding_dimension (int): Embeddings dimension for converting Token ids into dense Embeddings Vectors
            number_of_layers (int): Number of Decoder Layers in parrallel Decoder stacks
            number_of_heads (int): Number of Heads in Multi Head attention layer for Decoder stacks
            decoder_stacks (int): Numbber of Decoder Stacks serially connected together
            feed_forward_dimension (int): Dimension of the Feed forward layer
            dropout_rate (float): Dropout for reducing Attention over training
            training (bool): Whether the model is being trained or not
            
        Returns:
            None
        """
        super().__init__()
        self.number_of_tokens = number_of_tokens
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.training = training
        self.decoder_stacks = decoder_stacks

        if feed_forward_dimension is None:
            self.feed_forward_dimension = embedding_dimension * 4
        else:
            self.feed_forward_dimension = feed_forward_dimension

        self.dropout_rate = dropout_rate
        self.token_embedding = EmbeddingLayer(embedding_dimension, number_of_tokens)
        self.positional_encoding = PositionalEncodingLayer(embedding_dimension)
        self.layer_normalization = torch.nn.LayerNorm(embedding_dimension)
        self.decoder_stacks = torch.nn.ModuleList([DecoderStackLayer(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_layers,
            number_of_heads=number_of_heads,
            dropout_rate=dropout_rate,
            max_sequence_length=max_sequence_length,
            stack_index = i,
            max_stack = decoder_stacks - 1,
            training=training
        ).to('cuda',non_blocking=True) for i in range(decoder_stacks)])
        self.lm_head = LMHeadLayer(8*embedding_dimension, number_of_tokens).to(device='cuda',dtype=torch.float16)
    
    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        """Forward function for the model to predict next token probabilities
        
        Args:
            x (Tensor): Input Token Tensor for the model to use for prediction or training
            mask (Tensor): Mask Tensor for the model to use as Attention Masks
            
        Returns:
            Tensor: Output Tensor from the model with +1 context window
        """
        token_embeddings = self.token_embedding(x)
        positional_encoding = self.positional_encoding(token_embeddings)
        # positional_encoding_normalized = self.layer_normalization(positional_encoding)
        decoder_outputs = positional_encoding
        for decoder_stack in self.decoder_stacks:
            decoder_outputs = decoder_stack(decoder_outputs.to('cuda',non_blocking=True), mask.to('cuda',non_blocking=True))
            mask = torch.ones(mask.size(), device = 'cuda')
        lm_head_outputs = self.lm_head(decoder_outputs.to(dtype=torch.float16))
        return lm_head_outputs.to(device='cpu',dtype=torch.float32)
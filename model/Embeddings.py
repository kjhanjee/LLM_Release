"""Module for creating Embeddings layer class for the Transformer model
"""

# Standard imorts
import torch

class EmbeddingLayer(torch.nn.Module):
    """
    Module for creating Embeddings layer for the Transformer model
    """

    def __init__(self,embedding_dimension: int,number_of_tokens: int):
        """Init function to initialize class variables
        
        Args:
            embeddings_dimension (int): Dimensions of the embeddings vector created for each token in the input
            number_of_tokens (int): Tokenizer Vocabulary currently being predicted by the model
        
        Returns:
            None
        """
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(num_embeddings=number_of_tokens,embedding_dim=embedding_dimension)

    def forward(self, x: torch.Tensor):
        """Forward function for the embeddings layer
        
        Args: 
            x (Tensor): Input Tensor for the Embeddings layer. Shape expected Batch Size X Sequence Length
            
        Returns:
            Tensor: Output Tensor Embeddings for the Token Sequence
        """
        embedding = self.embedding_layer(x)
        return embedding
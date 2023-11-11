"""Module for Creating Feed Forward layer class to be utilized by the Decoder class
"""
# Standard Imports
import torch
from torch.cuda.amp.autocast_mode import custom_fwd
 
class FeedForwardLayer(torch.nn.Module):
    """Module for Creating Feed Forward layer to be utilized by the Decoder class
    """

    def __init__(self, embedding_dimension: int, feed_forward_dimension: int):
        """Init function to intialize the class variables
        
        Args:
            embedding_dimension (int): Embedding Layer Dimension for each token provided as input to the Transformer Model
            feed_forward_dimension (int): Feed Forward Dimensions for feature scaling the Token embeddings
        
        Returns:
            None
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.feed_forward_dimension = feed_forward_dimension
        self.linear_1 = torch.nn.Linear(embedding_dimension, feed_forward_dimension)
        self.linear_2 = torch.nn.Linear(feed_forward_dimension, embedding_dimension)

    @custom_fwd(cast_inputs=torch.float16)
    def forward(self, x: torch.Tensor):
        """Forward function for the layer
        
        Args:
            x (Tensor): Input Tensor from the Decoder Attention Layers
        
        Returns:
            None
        """
        out = self.linear_2(torch.nn.functional.relu(self.linear_1(x))) # Upsacling, then Relu and then downscaling again
        return out
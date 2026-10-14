import torch
import torch.nn as nn
from models.SimpleCNN import SimpleCNN
from models.SimpleViT import SimpleViT
import torch.nn.functional as F

class SimpleComposition(nn.Module):
    def __init__(self, CNN_hps, ViT_hps):
        super().__init__()
        self.cnn = SimpleCNN(**CNN_hps)
        self.vit = SimpleViT(**ViT_hps)
        
        
    def name(self):
        return "SimpleComposition"

    def forward(self, x):
        # TODO: Implement the forward method.
        '''
            Args:
                x (torch.Tensor): Input image tensor of shape (batch_size, 3, 32, 32).

            Returns:
                torch.Tensor: Logits of class predictions of shape (batch_size, 10), where 10 corresponds to the number of output classes.
        '''
        # raise NotImplementedError()
        feat = self.cnn.forward_map(x)   # (B, C, 4, 4)
        return self.vit(feat)
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    # TODO: Implement Simple CNN. You are allowed to use two conv layers (and pooling layers) and two linear layers.
    # NOTE: forward_map is used in Part 2.
    
    def __init__(self, conv_dim1, conv_dim2, fc_dim, dropout_rate):
        '''
        Args:
            conv_dim1 (int): Number of output channels for the first convolutional layer. 
            conv_dim2 (int): Number of output channels for the second convolutional layer.
            fc_dim (int): Number of channels in fully connected layer. 
            dropout_rate (float): Dropout rate to apply to prevent overfitting. 
        '''
        # raise NotImplementedError()

        super().__init__()
        self.conv1 = nn.Conv2d(3, conv_dim1, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(conv_dim1, conv_dim2, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(conv_dim2 * 4 * 4, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 10)
        
    def name(self):
        return "SimpleCNN"

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, 10), containing the logits for class predictions.
        '''
        # raise NotImplementedError()

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def forward_map(self, x):
        '''
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).
                
            Returns:
                torch.Tensor: feature maps of shape (batch_size, hidden_channel, hidden_h, hidden_w).
        '''
        # raise NotImplementedError()
    
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)
        return x
    

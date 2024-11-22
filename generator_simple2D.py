import torch
from torch import nn

def conv_block_2d(in_channels, out_channels):
    """
    A ConvBlock consisting of two 3x3 convolutional layers with stride 1 and Tanh activation,
    followed by a 1x1 convolutional layer with Tanh activation for 2D inputs.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.Tanh(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.Tanh(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
        nn.Tanh()
    )

class GeneratorSimple2D(nn.Module):
    def __init__(self):
        super(GeneratorSimple2D, self).__init__()
        
        # First ConvBlock with 64 feature maps
        self.conv_block1 = conv_block_2d(1, 64)
        
        # Second ConvBlock with 128 feature maps
        self.conv_block2 = conv_block_2d(64, 128)
        self.conv_block3 = conv_block_2d(128, 256)
        self.conv_block5 = conv_block_2d(256, 128)
        
        # Feature map reduction module
        self.feature_map_reduction = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),  # Reduces to 1 channel for Mel spectrogram
            nn.Tanh()
        )
        
    def forward(self, x):
        residual = x  # Save input as residual
        
        # Pass through ConvBlocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block5(x)
        
        # Reduce feature maps to match output dimensions
        x = self.feature_map_reduction(x)
        
        # Add residual (optional based on use case)
        x = x + residual
        
        return x

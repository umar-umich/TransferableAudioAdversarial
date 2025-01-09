### GeneratorSimple Class Code ###
import torch
from torch import nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels):
    """
    A ConvBlock consisting of two 3x3 convolutional layers with stride 1 and Swish activation,
    followed by a 1x1 convolutional layer with Swish activation.
    """
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.SiLU(),  # Swish activation
        nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.SiLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1),
        nn.SiLU()
    )

class GeneratorSimple(nn.Module):
    def __init__(self):
        super(GeneratorSimple, self).__init__()
        
        # First ConvBlock with 64 feature maps
        self.conv_block1 = conv_block(1, 64)
        
        # Second ConvBlock with 128 feature maps
        self.conv_block2 = conv_block(64, 128)
        self.conv_block3 = conv_block(128, 256)
        # self.conv_block4 = conv_block(256, 256)
        self.conv_block5 = conv_block(256, 128)
        
        # Feature map reduction module
        self.feature_map_reduction = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1),  # Reduces to 1 channel for audio
            nn.Tanh()  # Ensures output is in range [-1, 1]
        )
        
        # Residual connection scaling
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable scaling parameter

    def forward(self, x):
        residual = x  # Save input as residual
        
        # Pass through ConvBlocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block5(x)
        
        # Reduce feature maps to match output dimensions
        x = self.feature_map_reduction(x)
        
        # Add scaled residual connection
        x = self.alpha * residual + x


        # # Smoothing to mitigate high-frequency artifacts
        # x = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        
        # High-pass filtering to suppress zero Hz artifacts
        # x_mean = torch.mean(x, dim=-1, keepdim=True)  # Compute DC offset (mean along time dimension)
        # x = x - x_mean  # Remove low-frequency offset

        x = torch.clamp(x, -1, 1)

        return x

if __name__ == '__main__':
    G = GeneratorSimple().cuda()

    # Test the generator with audio input
    inputs = torch.rand(24, 1, 64600).cuda()  # Batch size: 24, Single channel, Length: 64600
    outputs = G(inputs)

    print("Output shape:", outputs.shape)


# # Tanh generator 
# ### GeneratorSimple Class Code ###
# import torch
# from torch import nn
# import torch.nn.functional as F

# def conv_block(in_channels, out_channels):
#     """
#     A ConvBlock consisting of two 3x3 convolutional layers with stride 1 and Swish activation,
#     followed by a 1x1 convolutional layer with Swish activation.
#     """
#     return nn.Sequential(
#         nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#         nn.Tanh(),  # Swish activation
#         nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#         nn.Tanh(),
#         nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1),
#         nn.Tanh()
#     )

# class GeneratorSimple(nn.Module):
#     def __init__(self):
#         super(GeneratorSimple, self).__init__()
        
#         # First ConvBlock with 64 feature maps
#         self.conv_block1 = conv_block(1, 64)
        
#         # Second ConvBlock with 128 feature maps
#         self.conv_block2 = conv_block(64, 128)
#         self.conv_block3 = conv_block(128, 256)
#         # self.conv_block4 = conv_block(256, 256)
#         self.conv_block5 = conv_block(256, 128)
        
#         # Feature map reduction module
#         self.feature_map_reduction = nn.Sequential(
#             nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1),  # Reduces to 1 channel for audio
#             nn.Tanh()  # Ensures output is in range [-1, 1]
#         )
        
#         # Residual connection scaling
#         self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable scaling parameter

#     def forward(self, x):
#         residual = x  # Save input as residual
        
#         # Pass through ConvBlocks
#         x = self.conv_block1(x)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x)
#         x = self.conv_block5(x)
        
#         # Reduce feature maps to match output dimensions
#         x = self.feature_map_reduction(x)
        
#         # Add scaled residual connection
#         x = self.alpha * residual + x
#         # x = residual + x


#         # # # Smoothing to mitigate high-frequency artifacts
#         # # x = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        
#         # # High-pass filtering to suppress zero Hz artifacts
#         x_mean = torch.mean(x, dim=-1, keepdim=True)  # Compute DC offset (mean along time dimension)
#         x = x - x_mean  # Remove low-frequency offset

#         # x = torch.clamp(x, -1, 1)

#         return x

# if __name__ == '__main__':
#     G = GeneratorSimple().cuda()

#     # Test the generator with audio input
#     inputs = torch.rand(24, 1, 64600).cuda()  # Batch size: 24, Single channel, Length: 64600
#     outputs = G(inputs)

#     print("Output shape:", outputs.shape)



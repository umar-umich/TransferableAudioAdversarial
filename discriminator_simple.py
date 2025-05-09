import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import SyncBatchNorm  # Import SyncBatchNorm


def block1_1d(in_channels, out_channels):
    """
    Block 1: Convolution + BatchNorm + Tanh + MaxPooling
    """
    layers = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3),
        SyncBatchNorm(out_channels),
        nn.Tanh(),
        nn.MaxPool1d(kernel_size=3, stride=2)
    )
    return layers

def block2_1d(in_channels, out_channels):
    """
    Block 2: Convolution + BatchNorm + Tanh + MaxPooling
    """
    layers = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding=2),
        SyncBatchNorm(out_channels),
        nn.Tanh(),
        nn.MaxPool1d(kernel_size=3, stride=2)
    )
    return layers

def block3_1d(in_channels, out_channels):
    """
    Block 3: 1x1 Convolution + BatchNorm + Tanh + AvgPooling
    """
    layers = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1),
        SyncBatchNorm(out_channels),
        nn.Tanh(),
        nn.AvgPool1d(kernel_size=5, stride=2)
    )
    return layers

def linear_block(in_features, out_features, activation="tanh"):
    """
    Fully connected block with optional activation (Tanh/Sigmoid)
    """
    layers = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=out_features),
        nn.Tanh() if activation == "tanh" else nn.Sigmoid()
    )
    return layers

class DiscriminatorSimple(nn.Module):
    def __init__(self):
        super(DiscriminatorSimple, self).__init__()
        
        # Constrained-CNN weight
        self.const_weight = nn.Parameter(torch.randn(size=[1, 1, 5]), requires_grad=True)
        
        # Convolutional blocks
        self.conv1 = block1_1d(1, 64)
        self.conv2 = block2_1d(64, 64)
        self.conv3 = block2_1d(64, 64)
        self.conv4 = block3_1d(64, 64)

        # Fully connected layers (adjusted dynamically during runtime)
        # self.fc1 = None  # Placeholder for the first fully connected layer
        self.fc1 = linear_block(47808, 256, activation="tanh")      # 47808 for ssdnet,  32064 for aasist

        self.fc2 = linear_block(256, 128, activation="tanh")
        self.fc3 = linear_block(128, 1, activation="sigmoid")

        # Initialize weights
        self.init_weight()

    def normalized_F(self):
        """
        Normalize the constrained-CNN weights without in-place operations.
        """
        # with torch.no_grad():  # Temporarily disable gradient tracking
        central_pixel = self.const_weight[:, 0, 2]
        sumed = self.const_weight.sum(dim=2) - central_pixel
        norm_weights = self.const_weight / sumed.unsqueeze(-1)
        norm_weights[:, 0, 2] = -1.0
        self.const_weight = nn.Parameter(norm_weights.clone())
        # self.const_weight = norm_weights.clone()  # Create a new tensor with updated weights
        # self.const_weight.copy_(norm_weights)  # Safely update the parameter


    # def normalized_F(self):
    #     """
    #     Normalize the constrained-CNN weights.
    #     """
    #     central_pixel = self.const_weight.data[:, 0, 2]
    #     sumed = self.const_weight.data.sum(dim=2) - central_pixel
    #     self.const_weight.data /= sumed.unsqueeze(-1)
    #     self.const_weight.data[:, 0, 2] = -1.0
    
    def init_weight(self):
        """
        Initialize weights of the layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, inputs):
        """
        Forward pass of the discriminator.
        """
        # Constrained-CNN
        self.normalized_F()
        outputs = F.conv1d(inputs, self.const_weight)
        
        # Pass through convolutional layers
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        
        # # Dynamically set the fully connected layer dimensions
        # if self.fc1 is None:
        #     flattened_size = outputs.shape[1] * outputs.shape[2]
        #     self.fc1 = linear_block(flattened_size, 200, activation="tanh").to(outputs.device)

        # Flatten and pass through fully connected layers
        outputs = torch.flatten(outputs, 1)
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)

        return outputs


if __name__ == "__main__":
    D = DiscriminatorSimple().cuda()

    # Test the discriminator with audio input
    inputs = torch.rand(20, 1, 64600).cuda()  # Batch size: 20, Single channel, Length: 64600
    outputs = D(inputs)

    print("Output shape:", outputs.shape)

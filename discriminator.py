import torch
import torch.nn as nn
import torch.nn.functional as F

def block1_1d(in_channels, out_channels):
    layers = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm1d(out_channels),
        nn.Tanh(),
        nn.MaxPool1d(kernel_size=3, stride=2)
    )
    return layers

def block2_1d(in_channels, out_channels):
    layers = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm1d(out_channels),
        nn.Tanh(),
        nn.MaxPool1d(kernel_size=3, stride=2)
    )
    return layers

def block3_1d(in_channels, out_channels):
    layers = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm1d(out_channels),
        nn.Tanh(),
        nn.AvgPool1d(kernel_size=3, stride=2)
    )
    return layers

def block4(in_features, out_features):
    layers = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=out_features),
        nn.Tanh()
    )
    return layers

def block5(in_features, out_features):
    layers = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=out_features),
        nn.Sigmoid()
    )
    return layers

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.const_weight = nn.Parameter(torch.randn(size=[1, 1, 5]), requires_grad=True)
        
        self.conv1 = block1_1d(1, 96)
        self.conv2 = block2_1d(96, 64)
        self.conv3 = block2_1d(64, 64)
        self.conv4 = block3_1d(64, 128)

        # Initialize fully connected layers with a placeholder feature size
        self.fc1 = None
        self.fc2 = block4(200, 200)
        self.fc3 = block5(200, 1)

        self.init_weight()

    def normalized_F(self):
        central_pixel = (self.const_weight.data[:, 0, 2])
        sumed = self.const_weight.data.sum(dim=2) - central_pixel
        self.const_weight.data /= sumed.unsqueeze(-1)
        self.const_weight.data[:, 0, 2] = -1.0
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, inputs):
        # Constrained-CNN
        self.normalized_F()
        outputs = F.conv1d(inputs, self.const_weight)
        # CNN
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        
        # Dynamically set the fully connected layer dimensions
        if self.fc1 is None:
            flattened_size = outputs.shape[1] * outputs.shape[2]
            self.fc1 = block4(flattened_size, 200).to(outputs.device)

        outputs = torch.flatten(outputs, 1)
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)

        return outputs


if __name__ == "__main__":
    D = Discriminator()

    inputs = torch.rand(20, 1, 64600)  # Batch size: 20, Single channel, Length: 64600
    outputs = D(inputs)

    print(outputs.shape)

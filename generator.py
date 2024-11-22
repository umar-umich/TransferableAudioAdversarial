import torch
from torch import nn

def conv_block_1d(in_channels, out_channels):
    layers = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.Tanh(),
        nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.Tanh(),
        nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.Tanh()
    )
    return layers

class Generator(nn.Module):
    def __init__(self, num_layers=4, num_features=128):
        super(Generator, self).__init__()
        self.num_layers = num_layers

        self.conv = conv_block_1d(1, num_features)  # Input channel is 1 for audio

        self.dconv = nn.Sequential(
            nn.Conv1d(num_features, num_features, kernel_size=3, stride=2, padding=1),
            nn.Tanh()
        )
        self.uconv = nn.ConvTranspose1d(
            num_features, num_features, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        conv_layers = []
        deconv_layers = []

        # Convolutional layers
        for _ in range(num_layers):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(num_features, num_features, kernel_size=3, padding=1),
                    nn.Tanh()
                )
            )

        # Deconvolutional layers
        for _ in range(num_layers - 1):
            deconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(num_features, num_features, kernel_size=3, padding=1),
                    nn.Tanh()
                )
            )

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)

        self.uconv1 = nn.ConvTranspose1d(num_features, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        residual = x

        x = self.conv(x)
        conv_feats = []

        for i in range(self.num_layers):
            x = self.conv_layers[i](x)

            if i % 4 == 0:
                x = self.dconv(x)
                conv_feats.append(x)

        conv_feats_idx = len(conv_feats) - 1
        for i in range(self.num_layers - 1):
            x = self.deconv_layers[i](x)

            if i == 0:
                x = x + conv_feats[conv_feats_idx]
                x = self.Tanh(x)
                conv_feats_idx -= 1
            elif i % 4 == 0:
                x = self.uconv(x)
                x = x + conv_feats[conv_feats_idx]
                x = self.Tanh(x)
                conv_feats_idx -= 1

        x = self.uconv1(x) + residual
        x = self.Tanh(x)

        return x

if __name__ == '__main__':
    from torchsummary import summary
    G = Generator().cuda()

    # Test the generator with audio input
    inputs = torch.rand(24, 1, 64600).cuda()  # Batch size: 32, Single channel, Length: 64600
    outputs = G(inputs)

    print(outputs.shape)

    # Summary
    # summary(G, (1, 64600))

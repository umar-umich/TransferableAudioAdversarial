import torch
import torch.nn as nn
import torch.nn.functional as F

# Self-ONN Layer (Custom Layer Based on Your Diagram)
class SelfONNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(SelfONNLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder (Downsampling)
        self.encoder = nn.Sequential(
            SelfONNLayer(1, 64, kernel_size=5, stride=2),  # 96000 -> 48000
            SelfONNLayer(64, 128, kernel_size=5, stride=2),  # 48000 -> 24000
            SelfONNLayer(128, 256, kernel_size=5, stride=2),  # 24000 -> 12000
            SelfONNLayer(256, 512, kernel_size=5, stride=2),  # 12000 -> 6000
        )

        # Decoder (Upsampling)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 6000 -> 12000
            SelfONNLayer(512, 256, kernel_size=5, stride=1),

            nn.Upsample(scale_factor=2, mode="nearest"),  # 12000 -> 24000
            SelfONNLayer(256, 128, kernel_size=5, stride=1),

            nn.Upsample(scale_factor=2, mode="nearest"),  # 24000 -> 48000
            SelfONNLayer(128, 64, kernel_size=5, stride=1),

            nn.Upsample(scale_factor=2, mode="nearest"),  # 48000 -> 96000
            SelfONNLayer(64, 1, kernel_size=5, stride=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            SelfONNLayer(1, 16, kernel_size=4, stride=8),
            SelfONNLayer(16, 32, kernel_size=4, stride=8),
            SelfONNLayer(32, 64, kernel_size=4, stride=8),
            SelfONNLayer(64, 128, kernel_size=4, stride=4),
            SelfONNLayer(128, 256, kernel_size=4, stride=4),
            SelfONNLayer(256, 1, kernel_size=3, stride=4),  # Output shape: [64, 1, small_time_steps]
        )

        # Global Average Pooling to reduce last dimension
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Ensures output shape [batch, 1, 1]

    def forward(self, x):
        x = self.model(x)  # Shape: [batch_size, 1, small_time_steps]
        x = self.global_avg_pool(x)  # Shape: [batch_size, 1, 1]
        return x.squeeze(2)  # Final shape: [batch_size, 1]


# Testing the models
if __name__ == "__main__":
    batch_size = 64
    input_audio = torch.randn(batch_size, 1, 96000)  # Simulated audio input

    generator = Generator()
    discriminator = Discriminator()

    generated_output = generator(input_audio)
    print("Generated output shape:", generated_output.shape)  # Expected: [64, 1, 96000]

    real_output = discriminator(input_audio)
    fake_output = discriminator(generated_output)

    print("Discriminator output for real input shape:", real_output.shape)  # Expected: [64, 1]
    print("Discriminator output for fake input shape:", fake_output.shape)  # Expected: [64, 1]

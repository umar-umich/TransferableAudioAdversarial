import torch
import torch.nn as nn

class Generator(nn.Module):
    """Restoration Generator Network using U-Net style architecture for batchx1x96000 input."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1), nn.Tanh(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1), nn.InstanceNorm1d(128), nn.Tanh(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1), nn.InstanceNorm1d(256), nn.Tanh(),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1), nn.InstanceNorm1d(512), nn.Tanh()
        )
        
        self.middle = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=4, stride=2, padding=1), nn.Tanh(),
            nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2, padding=1), nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1), nn.InstanceNorm1d(256), nn.Tanh(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1), nn.InstanceNorm1d(128), nn.Tanh(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1), nn.InstanceNorm1d(64), nn.Tanh(),
            nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1)  # Final convolutional layer
        )
    
    def forward(self, x):
        e1 = self.encoder(x)
        middle = self.middle(e1)
        return self.decoder(middle)

class Discriminator(nn.Module):
    """Discriminator Network for Restoration GAN with 1 channel input for batchx1x96000 input."""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1), nn.Tanh(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1), nn.InstanceNorm1d(128), nn.Tanh(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1), nn.InstanceNorm1d(256), nn.Tanh(),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1), nn.InstanceNorm1d(512), nn.Tanh(),
            nn.Conv1d(512, 1, kernel_size=4, stride=2, padding=1)  # Final convolutional layer
        )
        
        # Add a global average pooling layer to reduce to (batch_size, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.model(x)  # Pass through the convolutions
        x = self.pool(x)   # Apply global average pooling
        return x.view(-1, 1)  # Flatten to (batch_size, 1)

def main():
    # Create a sample input tensor with batch size 128 and sequence length 96000
    batch_size = 64
    input_tensor = torch.randn(batch_size, 1, 96000)

    # Instantiate the Generator and Discriminator models
    generator = Generator()
    discriminator = Discriminator()

    # Forward pass through the generator
    generated_output = generator(input_tensor)
    print(f"Generated output shape: {generated_output.shape}")  # Expected: (128, 1, 96000)

    # Forward pass through the discriminator
    discriminator_output_real = discriminator(input_tensor)  # For real input
    print(f"Discriminator output for real input shape: {discriminator_output_real.shape}")  # Expected: (128, 1)

    discriminator_output_fake = discriminator(generated_output)  # For generated input
    print(f"Discriminator output for generated input shape: {discriminator_output_fake.shape}")  # Expected: (128, 1)

if __name__ == "__main__":
    main()

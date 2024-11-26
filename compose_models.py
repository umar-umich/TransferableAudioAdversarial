

import torch
from torch import nn
from models.rawnet.RawNet3 import RawNet3
from models.rawnet.RawNetBasicBlock import Bottle2neck


def get_rawnet3():
    rawnet_model = RawNet3(
        Bottle2neck,
        model_scale=8,
        context=True,
        summed=True,
        encoder_type="ECA",
        nOut=256,
        out_bn=False,
        sinc_stride=10,
        log_sinc=True,
        norm_sinc="mean",
        grad_mult=1,
    )
    rawnet_model.load_state_dict(
        torch.load(
            "./weights/rawnet_3/model.pt",
            map_location=lambda storage, loc: storage, weights_only=True
        )["model"]
    )
    # rawnet_model = rawnet_model.to(device)  # Move model to the appropriate device

    rawnet_model.eval()
    print("RawNet3 initialised & weights loaded!")

    return rawnet_model

# Define the modified RawNet with a trainable FC layer
class RawNetWithFC(nn.Module):
    def __init__(self, embedding_dim=256, num_classes=2):
        super(RawNetWithFC, self).__init__()
        self.rawnet = get_rawnet3()
        for param in self.rawnet.parameters():  # Freeze all RawNet layers
            param.requires_grad = False
        self.fc = nn.Linear(embedding_dim, num_classes)  # Trainable FC layer

    def forward(self, x):
        x = self.rawnet(x)  # Get embeddings
        x = self.fc(x)  # Pass through FC layer
        return x
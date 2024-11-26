

import json
import torch
from torch import nn
import yaml
from models.aasist.AASIST import Model_ASSIST
from models.rawboost.RawBoost import RawNet
from models.rawnet.RawNet3 import RawNet3
from models.rawnet.RawNetBasicBlock import Bottle2neck
from models.rsm1d.RSM1D import SSDNet1D


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
    # rawnet_model.load_state_dict(
    #     torch.load(
    #         "./weights/rawnet_3/model.pt",
    #         map_location=lambda storage, loc: storage, weights_only=True
    #     )["model"]
    # )
    # rawnet_model = rawnet_model.to(device)  # Move model to the appropriate device

    # rawnet_model.eval()
    print("RawNet3 initialised & weights loaded!")

    return rawnet_model

# Define the modified RawNet with a trainable FC layer
class RawNetWithFC(nn.Module):
    def __init__(self, embedding_dim=256, num_classes=2):
        super(RawNetWithFC, self).__init__()
        self.rawnet = get_rawnet3()
        # for param in self.rawnet.parameters():  # Freeze all RawNet layers
        #     param.requires_grad = False
        self.fc = nn.Linear(embedding_dim, num_classes)  # Trainable FC layer

    def forward(self, x):
        x = self.rawnet(x)  # Get embeddings
        x = self.fc(x)  # Pass through FC layer
        return x
    


def get_aasist(device):
    # Load the AASIST model
    with open("./models/aasist/AASIST.conf", "r") as f_json:
        assist_config = json.loads(f_json.read())
    model_config = assist_config["model_config"]
    # model_config = config["model_config"]

    # print(f'ASSIST Conf: {str(model_config)}')
    assist_model = Model_ASSIST(model_config)
    assist_model.load_state_dict(torch.load("./weights/AASIST.pth", map_location=device, weights_only=True))
    assist_model = assist_model.to(device)  # Move model to the appropriate device
    assist_model.eval()  # Set the model to evaluation mode
    return assist_model


def get_ssdnet(device):
    ssdnet_model = SSDNet1D()
    num_total_learnable_params = sum(i.numel() for i in ssdnet_model.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))

    check_point = torch.load("./weights/ssdnet/ssdnet_1.64.pth", map_location=device, weights_only=True)
    ssdnet_model.load_state_dict(check_point['model_state_dict'])
    ssdnet_model = ssdnet_model.to(device)  # Move model to the appropriate device
    ssdnet_model.eval()
    return ssdnet_model


def get_rawboost(device):
    with open("./models/rawboost/model_config_RawNet.yaml", 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)
    rawboost_model = RawNet(parser1['model'], device)
    rawboost_model.load_state_dict(torch.load("./weights/rawboost/Best_model.pth", map_location=device, weights_only=True))
    rawboost_model = rawboost_model.to(device)  # Move model to the appropriate device
    rawboost_model.eval()
    return rawboost_model

def get_rawnet(device):
    # Instantiate the model
    rawnet_model = RawNetWithFC(embedding_dim=256, num_classes=2)#.to(device)
    rawnet_model.load_state_dict(torch.load("./weights/rawnet_3/best_rawnet3.pth", map_location=device, weights_only=True))
    rawnet_model = rawnet_model.to(device)  # Move model to the appropriate device
    rawnet_model.eval()
    return rawnet_model


def get_wav2vec2_model(device):
    # load model and processor
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")  # facebook/wav2vec2-base-960h
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    # processor = processor.to(device)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return processor, model
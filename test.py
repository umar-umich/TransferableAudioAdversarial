import torch
from torch import nn
from torch.utils import data
# import timm
import argparse

import numpy as np
from sklearn.metrics import *
from tqdm import tqdm
# from natsort import natsort_keygen
from generator import Generator
from discriminator import Discriminator
from data_loader import DATAReader
import json
from generator_simple import GeneratorSimple
from models.aasist.AASIST import Model_ASSIST
from models.rawnet.RawNet3 import RawNet3
from models.rawnet.RawNetBasicBlock import Bottle2neck
from models.rsm1d.RSM1D import SSDNet1D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_aasist():
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
    rawnet_model = rawnet_model.to(device)  # Move model to the appropriate device

    rawnet_model.eval()
    print("RawNet3 initialised & weights loaded!")

    return rawnet_model

class RawNet3BinaryClassifier(nn.Module):
    def __init__(self, rawnet_model, feature_dim=256, num_classes=2):
        super(RawNet3BinaryClassifier, self).__init__()
        self.rawnet = rawnet_model
        # Append a fully connected layer for binary classification
        self.fc = nn.Linear(feature_dim, num_classes)  # Map 128 to 2 logits

    def forward(self, x):
        x = self.rawnet(x)
        x = self.fc(x)
        return x


# Load RawNet3 and wrap it
def get_rawnet3_binary_classifier():
    # Load RawNet3 as usual
    rawnet_model = get_rawnet3()
    # Wrap with a binary classifier
    binary_model = RawNet3BinaryClassifier(rawnet_model, feature_dim=256, num_classes=2)
    binary_model = binary_model.to(device)
    return binary_model


def get_ssdnet():
    ssdnet_model = SSDNet1D()
    num_total_learnable_params = sum(i.numel() for i in ssdnet_model.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))

    check_point = torch.load("./weights/ssdnet/ssdnet_1.64.pth", weights_only=True)
    ssdnet_model.load_state_dict(check_point['model_state_dict'])
    ssdnet_model = ssdnet_model.to(device)  # Move model to the appropriate device

    ssdnet_model.eval()
    print("SSDNet initialised & weights loaded!")

    return ssdnet_model


def cal_acc(model_name, y, x):
    # outputs = inception(x)
    outputs = {}
    if 'AASIST' in model_name:
        assist_model = get_aasist()
        outputs = assist_model(x.squeeze(1))
        print(f'Shape of AASIST output: {str(outputs[0].shape)}: {str(outputs[1].shape)}')
        predictions = outputs[1]
    elif 'RawNet3' in model_name:
        rawnet3 = get_rawnet3_binary_classifier()
        outputs = rawnet3(x.squeeze(1))
        print(f'Shape of rawnet output: {str(outputs.shape)}')
        predictions = outputs
    
    elif 'SSDNet' in model_name:
        ssdnet = get_ssdnet()
        # x = x.unsqueeze(1)  # Add channel dimension
        print(f'Shape of input : {str(x.shape)}')
        outputs = ssdnet(x)
        print(f'Shape of ssdnet output: {str(outputs.shape)}')
        predictions = outputs
        
    predictions = nn.Softmax(dim=-1)(predictions)
    _, y_ = torch.max(predictions, 1)

    acc = accuracy_score(y.cpu().numpy(), y_.cpu().numpy())

    return acc


def test(model_name):

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=12, help='')
    parser.add_argument('--num_workers', '-w', type=int, default=16, help='')

    args = parser.parse_args()
    print(args)

    test_dataset = DATAReader(args=args, split='TEST')
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # print('Device being used:', device)

    # G = nn.DataParallel(Generator())
    G = GeneratorSimple()


    checkpoint = torch.load("./CHECKPOINTS_2024-11-25-14-38-25_ssdnet_0.0001_10/generator_27.pth", map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']

    # Remove 'module.' prefix from keys
    # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    G.load_state_dict(state_dict)
    G = G.to(device)  # Move model to the appropriate device
    G.eval()
    real_acc, fake_acc, af_acc = [], [], []
    progress_bar = tqdm(test_loader, desc="Testing", unit="batch", leave=True)

    for test_sample in progress_bar:
        real = test_sample[0].unsqueeze(1).to(device, dtype=torch.float)
        forged = test_sample[2].unsqueeze(1).to(device, dtype=torch.float)

        y_real =  torch.zeros(real.shape[0]).to(device, dtype=torch.float)
        y_fake =  torch.ones(forged.shape[0]).to(device, dtype=torch.float)

        fake = G(forged)

        real_acc.append(cal_acc(model_name, y_real, real))
        fake_acc.append(cal_acc(model_name, y_fake, forged))
        af_acc.append(cal_acc(model_name, y_fake, fake))

    progress_bar.close()  # Ensure tqdm closes cleanly when done
    return 100*np.mean(real_acc), 100*np.mean(fake_acc), 100*np.mean(af_acc)

if __name__ == '__main__':
    model_name = 'AASIST'  # RawNet3, AASIST, SSDNet
    # test(model_name)
    r_acc, f_acc, af_acc = test(model_name)
    print('[Test] [[Acc: %.2f, %.2f, %.2f]'% (r_acc, f_acc, af_acc))

    # Example usage
    # binary_classifier = get_rawnet3_binary_classifier()

    # # Input tensor: batch of 16 samples
    # x = torch.randn(1, 1, 64600).to(device)

    # # Get binary class logits
    # outputs = binary_classifier(x.squeeze(1))
    # print(f"Shape of binary classifier output: {outputs.shape}")

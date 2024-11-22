import torch
from torch import nn
from torch.utils import data
# import timm
import argparse

import numpy as np
from sklearn.metrics import *
# from natsort import natsort_keygen
from generator import Generator
from discriminator import Discriminator
from data_loader import DATAReader
import json
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
    assist_model.load_state_dict(torch.load("./weights/AASIST.pth", map_location=device))
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
            map_location=lambda storage, loc: storage,
        )["model"]
    )
    rawnet_model = rawnet_model.to(device)  # Move model to the appropriate device

    rawnet_model.eval()
    print("RawNet3 initialised & weights loaded!")

    return rawnet_model


def get_ssdnet():
    ssdnet_model = SSDNet1D()
    num_total_learnable_params = sum(i.numel() for i in ssdnet_model.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))

    check_point = torch.load("./weights/ssdnet/ssdnet_1.64.pth")
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
        rawnet3 = get_rawnet3()
        outputs = rawnet3(x.squeeze(1))
        print(f'Shape of rawnet output: {str(outputs.shape)}')
        predictions = outputs[1]
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
    parser.add_argument('--batch_size', '-b', type=int, default=24, help='')
    parser.add_argument('--num_workers', '-w', type=int, default=16, help='')

    args = parser.parse_args()
    print(args)

    test_dataset = DATAReader(args=args, split='TEST')
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # print('Device being used:', device)

    # G = nn.DataParallel(Generator())
    G = Generator()


    checkpoint = torch.load("./CHECKPOINTS/generator_14.pth", map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']

    # Remove 'module.' prefix from keys
    # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    G.load_state_dict(state_dict)
    G = G.to(device)  # Move model to the appropriate device
    G.eval()
    real_acc, fake_acc, af_acc = [], [], []
    for test_sample in test_loader:
        real = test_sample[0].unsqueeze(1).to(device, dtype=torch.float)
        forged = test_sample[2].unsqueeze(1).to(device, dtype=torch.float)

        y_real =  torch.ones(real.shape[0]).to(device, dtype=torch.float)
        y_fake =  torch.zeros(forged.shape[0]).to(device, dtype=torch.float)

        fake = G(forged)

        real_acc.append(cal_acc(model_name, y_real, real))
        fake_acc.append(cal_acc(model_name, y_fake, forged))
        af_acc.append(cal_acc(model_name, y_fake, fake))

    return 100*np.mean(real_acc), 100*np.mean(fake_acc), 100*np.mean(af_acc)

if __name__ == '__main__':
    model_name = 'SSDNet'  # RawNet3, AASIST, 
    # test(model_name)
    r_acc, f_acc, af_acc = test(model_name)
    print('[Test] [[Acc: %.2f, %.2f, %.2f]'% (r_acc, f_acc, af_acc))

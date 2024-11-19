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

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-b', type=int, default=32, help='')
parser.add_argument('--num_workers', '-w', type=int, default=16, help='')

args = parser.parse_args()
print(args)

test_dataset = DATAReader(args=args, split='TEST')
test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Device being used:', device)

# G = nn.DataParallel(Generator())
G = Generator()


checkpoint = torch.load("./CHECKPOINTS/generator_14.pth", map_location=device, weights_only=False)
state_dict = checkpoint['state_dict']

# Remove 'module.' prefix from keys
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
G.load_state_dict(new_state_dict)
G = G.to(device)  # Move model to the appropriate device


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

def cal_acc(model_name, y, x):
    # outputs = inception(x)
    outputs = {}
    if 'AASIST' in model_name:
        outputs = assist_model(x.squeeze(1))
    outputs = nn.Softmax(dim=-1)(outputs[1])
    _, y_ = torch.max(outputs, 1)

    acc = accuracy_score(y.cpu().numpy(), y_.cpu().numpy())

    return acc


def test(model_name):
    G.eval()
    real_acc, fake_acc, af_acc = [], [], []
    for test_sample in test_loader:
        real = test_sample[0].to(device, dtype=torch.float)
        forged = test_sample[2].unsqueeze(1).to(device, dtype=torch.float)

        y_real =  torch.ones(real.shape[0]).to(device, dtype=torch.float)
        y_fake =  torch.zeros(forged.shape[0]).to(device, dtype=torch.float)

        fake = G(forged)

        real_acc.append(cal_acc(model_name, y_real, real))
        fake_acc.append(cal_acc(model_name, y_fake, forged))
        af_acc.append(cal_acc(model_name, y_fake, fake))

    return 100*np.mean(real_acc), 100*np.mean(fake_acc), 100*np.mean(af_acc)

if __name__ == '__main__':
    model_name = 'AASIST'  # RawNet3, AASIST
    # test(model_name)
    r_acc, f_acc, af_acc = test(model_name)
    print('[Test] [[Acc: %.2f, %.2f, %.2f]'% (r_acc, f_acc, af_acc))

import os
import os.path as osp
import sys
import argparse
import numpy as np
from torch.utils import data
from torch.utils.data import ConcatDataset

from torch.optim.lr_scheduler import StepLR
import torch
from torch import nn
# import timm
from sklearn.metrics import *
# from natsort import natsort_keygen
# from generator import Generator
# from discriminator import Discriminator
from generator_simple import GeneratorSimple
from discriminator_simple import DiscriminatorSimple
from data_loader import DATAReader
import json
from models.aasist.AASIST import Model_ASSIST
from models.rsm1d.RSM1D import SSDNet1D
from visualize import compare_audio_samples
from tqdm import tqdm



# natsort_key = natsort_keygen(key = lambda y: y.lower())

parser = argparse.ArgumentParser()
parser.add_argument('--root', '-rt', type=str, default='../DATASETS/DTIM', help='')
parser.add_argument('--nEpochs', '-epoch', type=int, default=30, help='')
parser.add_argument('--batch_size', '-b', type=int, default=12, help='')
parser.add_argument('--num_workers', '-w', type=int, default=16, help='')
parser.add_argument('--lr', '-lr', type=float, default=0.0001, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0], help='')
args = parser.parse_args()
print(args)

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Device being used:', device)

G = GeneratorSimple().to(device)
D = DiscriminatorSimple().to(device)

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


ssdnet_model = SSDNet1D()
num_total_learnable_params = sum(i.numel() for i in ssdnet_model.parameters() if i.requires_grad)
print('Number of learnable params: {}.'.format(num_total_learnable_params))

check_point = torch.load("./weights/ssdnet/ssdnet_1.64.pth", map_location=device, weights_only=True)
ssdnet_model.load_state_dict(check_point['model_state_dict'])
ssdnet_model = ssdnet_model.to(device)  # Move model to the appropriate device
ssdnet_model.eval()

train_dataset = DATAReader(args=args, split='TRAIN')
dev_dataset = DATAReader(args=args, split='DEV')  # Add a development dataset
train_dataset = ConcatDataset([train_dataset, dev_dataset])

train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

test_dataset = DATAReader(args=args, split='TEST')
test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

perceptual_loss = nn.MSELoss() # nn.L1Loss()
adversarial_loss = nn.BCELoss()
classifiation_loss = nn.CrossEntropyLoss()

optimizer_G = torch.optim.Adam(G.parameters(), lr = args.lr, betas = (0.9, 0.999), eps = 0.00000001)
optimizer_D = torch.optim.SGD(D.parameters(), lr = args.lr)

scheduler_G = StepLR(optimizer_G, step_size=10, gamma=0.9)
scheduler_D = StepLR(optimizer_D, step_size=10, gamma=0.9)

def sLoss(x, y):
    # logits = assist_model(x.squeeze(1))[1]  # Use the first item of the tuple
    logits = ssdnet_model(x)
    s_loss = classifiation_loss(logits, y.to(dtype=torch.long))
    # s_loss = classifiation_loss(assist_model(x.squeeze(1)), y.to(dtype=torch.long)) #+ classifiation_loss(inception(x), y.to(dtype=torch.long)) + \
             #classifiation_loss(mobilent(x), y.to(dtype=torch.long)) + classifiation_loss(resnet(x), y.to(dtype=torch.long)) + \
            #classifiation_loss(xception(x), y.to(dtype=torch.long))
    return s_loss

def train(epoch):
    g_losses = []
    c_losses = []

    d_losses = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=True)
    for index, train_sample in enumerate(progress_bar):
    # for index, train_sample in enumerate(train_loader):
        real = train_sample[0].unsqueeze(1).to(device, dtype=torch.float)  # remove .unsqueeze(1) for aasist
        forged = train_sample[2].unsqueeze(1).to(device, dtype=torch.float)
        
        # print(f'shape of real {real.shape}')
        # print(f'shape of forged {forged.shape}')

        y_real =  torch.zeros(real.shape[0]).to(device, dtype=torch.float)
        y_fake =  torch.ones(forged.shape[0]).to(device, dtype=torch.float)
        
        # ========================Generator==============================
        optimizer_G.zero_grad()

        fake = G(forged)

        if index == 0:
            real_audio = forged[0].detach()  # Select first sample of forged audio
            fake_audio = fake[0].detach()   # Corresponding generated audio

            # Plot and compare
            compare_audio_samples(real_audio, fake_audio,epoch, sr=16000)


        per_loss = perceptual_loss(forged, fake)
        adv_loss = adversarial_loss(y_real, D(fake).squeeze().detach())
        c_loss = sLoss(fake, y_real.to(dtype=torch.long))

        g_loss = per_loss + adv_loss + 0.01*c_loss # 0.0001

        g_loss.backward()
        optimizer_G.step()

        # =======================Discriminator============================
        optimizer_D.zero_grad()

        real_loss = adversarial_loss(D(real).squeeze(), y_real)
        fake_loss = adversarial_loss(D(fake.detach()).squeeze(), y_fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        #=================================================================

        g_losses.append(g_loss.item())
        c_losses.append(c_loss.item())
        d_losses.append(d_loss.detach().cpu().numpy())

        # Update tqdm progress bar with current losses
        progress_bar.set_postfix({
            "G_Loss": f"{np.mean(g_losses):.4f}",
            "C_Loss": f"{np.mean(c_losses):.4f}",
            "D_Loss": f"{np.mean(d_losses):.4f}"
        })

    progress_bar.close()  # Ensure tqdm closes cleanly when done
    return np.mean(g_losses), np.mean(c_losses), np.mean(d_losses)

def cal_acc(y, x):
    # outputs = inception(x)
    outputs = ssdnet_model(x)
    outputs = nn.Softmax(dim=-1)(outputs)
    _, y_ = torch.max(outputs, 1)

    acc = accuracy_score(y.cpu().numpy(), y_.cpu().numpy())

    return acc

def test(epoch=0):
    G.eval()
    real_acc, fake_acc, af_acc = [], [], []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=True)

    for test_sample in progress_bar:
        real = test_sample[0].unsqueeze(1).to(device, dtype=torch.float)
        forged = test_sample[2].unsqueeze(1).to(device, dtype=torch.float)

        y_real =  torch.zeros(real.shape[0]).to(device, dtype=torch.float)
        y_fake =  torch.ones(forged.shape[0]).to(device, dtype=torch.float)

        fake = G(forged)
        # real_audio = forged[0].detach()  # Select first sample of forged audio
        # fake_audio = fake[0].detach()   # Corresponding generated audio

        # Plot and compare
        # compare_audio_samples(real_audio, fake_audio, sr=16000)
        
        real_acc.append(cal_acc(y_real, real))
        fake_acc.append(cal_acc(y_fake, forged))
        af_acc.append(cal_acc(y_fake, fake))

    progress_bar.close()
    return 100*np.mean(real_acc), 100*np.mean(fake_acc), 100*np.mean(af_acc)

def main():
    print('Training on', 2*len(train_dataset), 'and validating on ', 2*len(test_dataset), 'samples.')
    for epoch in range(args.nEpochs):
        g_loss, c_loss, d_loss = train(epoch)

        scheduler_G.step()
        scheduler_D.step()

        print("[Train] [Epoch %d/%d], [LR: G=%f, D=%f], [C loss: %f], [D loss: %f], [G loss: %f]" % (
            epoch, args.nEpochs, scheduler_G.get_last_lr()[0],  scheduler_D.get_last_lr()[0], c_loss, d_loss, g_loss))

        checkpoints = {
            'epoch': epoch+1,
            'state_dict': G.state_dict()
        }
        torch.save(checkpoints, osp.join('CHECKPOINTS/generator_'+str(epoch+1)+'.pth'))

        r_acc, f_acc, af_acc = test(epoch)

        print('[Test] [Epoch %d/%d], [Acc: %.2f, %.2f, %.2f]'% (epoch, args.nEpochs, r_acc, f_acc, af_acc))

if __name__ == '__main__':
    main()
    # r_acc, f_acc, af_acc = test()
    # print('[Test] [[Acc: %.2f, %.2f, %.2f]'% (r_acc, f_acc, af_acc))


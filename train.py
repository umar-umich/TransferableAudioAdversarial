import os
import os.path as osp
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
from generator import Generator
from discriminator import Discriminator
# from compose_models import get_rawnet3
from compose_models import RawNetWithFC, get_inc_ssdnet, get_rawnet2, get_speech_to_text_model, get_ssdnet, get_wav2vec2_model
from generator_simple import GeneratorSimple
from discriminator_simple import DiscriminatorSimple
from data_loader import DATAReader
import json
from generator_simple2D import GeneratorSimple2D
from models.aasist.AASIST import Model_ASSIST
from models.rawboost.RawBoost import RawNet  # From RawBoost Repo
from models.rawnet.RawNetBasicBlock import Bottle2neck
from models.rsm1d.RSM1D import SSDNet1D
from models.rawnet.RawNet3 import RawNet3
from utils import batch_audio_to_mel, batch_mel_to_audio, get_transciption_loss, transcribe_audio, transcribe_s2t
from visualize import compare_audio_samples
from tqdm import tqdm
import datetime
import yaml


# natsort_key = natsort_keygen(key = lambda y: y.lower())

parser = argparse.ArgumentParser()
parser.add_argument('--root', '-rt', type=str, default='../DATASETS/DTIM', help='')
parser.add_argument('--nEpochs', '-epoch', type=int, default=50, help='')
parser.add_argument('--batch_size', '-b', type=int, default=16, help='')
parser.add_argument('--num_workers', '-w', type=int, default=16, help='')
parser.add_argument('--lr', '-lr', type=float, default=0.001, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0], help='')
parser.add_argument('--save_output', action='store_true', 
                    help="Flag to save the models/output to files for future use")
args = parser.parse_args()
print(args)


time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
save_dir_path = str(time_now)
# Ensure the directory exists


# gpu_devices = ','.join([str(id) for id in args.gpu_devices])
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Device being used:', device)

G = GeneratorSimple().to(device)
D = DiscriminatorSimple().to(device)



    # similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    # return 1 - similarity  # Convert similarity to loss

train_dataset = DATAReader(args=args, split='TRAIN')
# dev_dataset = DATAReader(args=args, split='DEV')  # Add a development dataset
# train_dataset = ConcatDataset([train_dataset, dev_dataset])

train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

test_dataset = DATAReader(args=args, split='TEST')
test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

perceptual_loss = nn.MSELoss() # nn.L1Loss()
# adversarial_loss = nn.BCELoss()
adversarial_loss = nn.BCEWithLogitsLoss()
classifiation_loss = nn.CrossEntropyLoss()

optimizer_G = torch.optim.Adam(G.parameters(), lr = args.lr, betas = (0.9, 0.999), eps = 0.00000001)
optimizer_D = torch.optim.SGD(D.parameters(), lr = args.lr)

scheduler_G = StepLR(optimizer_G, step_size=10, gamma=0.9)
scheduler_D = StepLR(optimizer_D, step_size=10, gamma=0.9)
s1_w = 0.0001
s2_w = 0.0001
# s3_w = 1   # 0.0001
t1_w = 5
t2_w = 5
# t2_we
model_name = 'ssdnet_inc_ssdnet'
cl_model1 = get_ssdnet(device)
cl_model2 = get_inc_ssdnet(device)
cl_model3 = get_rawnet2(device)
t_processor_1, t_model_1 = get_wav2vec2_model(device)
t_processor_2, t_model_2 = get_speech_to_text_model(device)
save_dir_path = f'{save_dir_path}_{model_name}_{s1_w}_{s2_w}_{t1_w}_{t2_w}'


scaler = torch.GradScaler(device)

def sLoss(x, y, model):
    # logits = assist_model(x.squeeze(1))[1]  # Use the first item of the tuple
    # x = x
    # print(f"Input shape: {x.shape}")
    # with torch.no_grad():
    logits = model(x)  # x.squeeze(1) for aasist 
    # print(f"Logits: {str(logits[0])}  :   {str(logits[1])}")
    s_loss = classifiation_loss(logits, y.to(dtype=torch.long))
    # s_loss = classifiation_loss(assist_model(x.squeeze(1)), y.to(dtype=torch.long)) #+ classifiation_loss(inception(x), y.to(dtype=torch.long)) + \
             #classifiation_loss(mobilent(x), y.to(dtype=torch.long)) + classifiation_loss(resnet(x), y.to(dtype=torch.long)) + \
            #classifiation_loss(xception(x), y.to(dtype=torch.long))
    return s_loss

def train(epoch):
    g_losses = []
    c1_losses = []
    c2_losses = []
    d_losses = []
    t1_losses = []
    t2_losses = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=True)

    for index, train_sample in enumerate(progress_bar):
        real = train_sample[0].unsqueeze(1).to(device, dtype=torch.float)
        forged = train_sample[2].unsqueeze(1).to(device, dtype=torch.float)

        y_real = torch.zeros(real.shape[0]).to(device, dtype=torch.float)
        y_fake = torch.ones(forged.shape[0]).to(device, dtype=torch.float)

        # ========================Generator==============================
        optimizer_G.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16):  # Enable Mixed Precision
            attacked = G(forged)

            forged_transciption1 = transcribe_audio(forged,t_processor_1,t_model_1,device)
            attacked_transciption1 = transcribe_audio(attacked,t_processor_1,t_model_1,device)

            forged_transciption2 = transcribe_s2t(forged,t_processor_2,t_model_2,device)
            attacked_transciption2 = transcribe_s2t(attacked,t_processor_2,t_model_2,device)

            if index == 0 and args.save_output:
                forged_audio = forged[0].detach()  # Select first sample of forged audio
                attacked_audio = attacked[0].detach()   # Corresponding generated audio

                # Plot and compare
                wav_dir_path = 'Wav_Plot_'+save_dir_path
                os.makedirs(wav_dir_path, exist_ok=True)
                compare_audio_samples(forged_audio, attacked_audio, forged_transciption1, attacked_transciption1, forged_transciption2, attacked_transciption2,epoch,wav_dir_path, sr=16000)


            t1_loss = get_transciption_loss(forged_transciption1,attacked_transciption1)
            t2_loss = get_transciption_loss(forged_transciption2,attacked_transciption2)

            per_loss = perceptual_loss(forged, attacked)
            adv_loss = adversarial_loss(y_real, D(attacked).squeeze().detach())
            c1_loss = sLoss(attacked, y_real.to(dtype=torch.long),cl_model1)
            c2_loss = sLoss(attacked, y_real.to(dtype=torch.long),cl_model2)
            # c3_loss = sLoss(attacked.squeeze(1), y_fake.to(dtype=torch.long),cl_model3)   # used fake label just to revert the label  for rawnet
            # c3_loss = np.float32(c3_loss.item())   # required for rawnet_2

            print(f"C losses: {c1_loss}, {c2_loss}, T1 loss: {t1_loss}, T2 loss: {t2_loss}")
            g_loss = per_loss + adv_loss + s1_w*c1_loss+ s2_w*c2_loss + t1_w*t1_loss + t2_w*t2_loss  # s2_w*c2_loss + 

        # Scale the loss and backpropagate
        scaler.scale(g_loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

        # =======================Discriminator============================
        optimizer_D.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16):  # Enable Mixed Precision
            real_loss = adversarial_loss(D(real).squeeze(), y_real)
            fake_loss = adversarial_loss(D(attacked.detach()).squeeze(), y_fake)
            d_loss = (real_loss + fake_loss) / 2

        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)
        scaler.update()

        g_losses.append(g_loss.item())
        c1_losses.append(c1_loss.item())  # +c2_loss.item()
        c2_losses.append(c2_loss.item())  # +c2_loss.item()
        d_losses.append(d_loss.detach().cpu().numpy())
        t1_losses.append(t1_loss.item())
        t2_losses.append(t2_loss.item())

        # Update tqdm progress bar with current losses
        progress_bar.set_postfix({
            "G_Loss": f"{np.mean(g_losses):.4f}",
            "C1_Loss": f"{np.mean(c1_losses):.4f}",
            "C2_Loss": f"{np.mean(c2_losses):.4f}",
            "D_Loss": f"{np.mean(d_losses):.4f}",
            "T1_Loss": f"{np.mean(t1_losses):.4f}",
            "T2_Loss": f"{np.mean(t2_losses):.4f}",
        })

    progress_bar.close()  # Ensure tqdm closes cleanly when done
    return np.mean(g_losses), np.mean(c1_losses),np.mean(c2_losses), np.mean(d_losses), np.mean(t1_losses), np.mean(t2_losses)

def cal_acc(y, x, model):
    # outputs = inception(x)
    # with torch.no_grad():
    outputs = model(x)   # (x.squeeze(1))[1] for aasist   # ssdnet_model, assist_model 
    outputs = nn.Softmax(dim=-1)(outputs)
    _, y_ = torch.max(outputs, 1)

    acc = accuracy_score(y.cpu().numpy(), y_.cpu().numpy())

    return acc

def test(epoch=0):
    G.eval()
    accuracies = {
        "cl_model1": {"real": [], "fake": [], "af": []},
        "cl_model2": {"real": [], "fake": [], "af": []},
        # "cl_model3": {"real": [], "fake": [], "af": []},
    }

    progress_bar = tqdm(test_loader, desc=f"[Test] Epoch {epoch}", unit="batch", leave=True)

    for test_sample in progress_bar:
        real = test_sample[0].unsqueeze(1).to(device, dtype=torch.float)
        forged = test_sample[2].unsqueeze(1).to(device, dtype=torch.float)

        y_real = torch.zeros(real.shape[0]).to(device, dtype=torch.float)
        y_fake = torch.ones(forged.shape[0]).to(device, dtype=torch.float)

        with torch.autocast(device_type='cuda', dtype=torch.float16):  # Enable Mixed Precision
            fake = G(forged)

            for model_name, model in zip(
                ["cl_model1", "cl_model2"], [cl_model1, cl_model2]
            ):
                
                if model_name in 'cl_model3':
                    accuracies[model_name]["real"].append(cal_acc(y_fake, real.squeeze(1), model))          # used fake label just to revert the label  for rawnet
                    accuracies[model_name]["fake"].append(cal_acc(y_real, forged.squeeze(1), model))   
                    accuracies[model_name]["af"].append(cal_acc(y_real, fake.squeeze(1), model))
                else:
                    accuracies[model_name]["real"].append(cal_acc(y_real, real, model))
                    accuracies[model_name]["fake"].append(cal_acc(y_fake, forged, model))
                    accuracies[model_name]["af"].append(cal_acc(y_fake, fake, model))

    progress_bar.close()

    # Compute mean accuracies
    results = {
        model_name: {
            metric: 100 * np.mean(values) for metric, values in acc_dict.items()
        }
        for model_name, acc_dict in accuracies.items()
    }

    return results



def main():
    print('Training on', 2*len(train_dataset), 'and validating on ', 2*len(test_dataset), 'samples.')

    for epoch in range(args.nEpochs):
        g_loss, c1_loss,c2_loss, d_loss, t1_loss, t2_loss = train(epoch)

        scheduler_G.step()
        scheduler_D.step()

        print("[Train] [Epoch %d/%d], [LR: G=%f, D=%f], [C1 loss: %f], [C2 loss: %f], [D loss: %f], [G loss: %f], [T1 loss: %f], [T2 loss: %f]" % (
            epoch, args.nEpochs, scheduler_G.get_last_lr()[0],  scheduler_D.get_last_lr()[0], c1_loss, c2_loss, d_loss, g_loss, t1_loss,t2_loss))


        results = test(epoch)

        # Extract results for each classifier
        r_acc, f_acc, af_acc = results["cl_model1"].values()
        r2_acc, f2_acc, af2_acc = results["cl_model2"].values()
        # r3_acc, f3_acc, af3_acc = results["cl_model3"].values()

        # Print the results
        print('[Test Cl_1] [Epoch %d/%d], [Acc: %.2f, %.2f, %.2f]' % (epoch, args.nEpochs, r_acc, f_acc, af_acc))
        print('[Test Cl_2] [Epoch %d/%d], [Acc: %.2f, %.2f, %.2f]' % (epoch, args.nEpochs, r2_acc, f2_acc, af2_acc))
        # print('[Test Cl_3] [Epoch %d/%d], [Acc: %.2f, %.2f, %.2f]' % (epoch, args.nEpochs, r3_acc, f3_acc, af3_acc))

        if args.save_output:
            checkpoints = {
                'epoch': epoch+1,
                'state_dict': G.state_dict()
            }
            
            checkpoint_dir_path = 'CHECKPOINTS_'+save_dir_path
            os.makedirs(checkpoint_dir_path, exist_ok=True)
            # Save the checkpoint
            torch.save(checkpoints, osp.join(checkpoint_dir_path, f'generator_{epoch+1}.pth'))

            # Save test accuracies to a file
            results_file = osp.join(checkpoint_dir_path, 'test_accuracies.txt')
            with open(results_file, 'a') as f:
                f.write(f"Cl1: Epoch {epoch + 1}/{args.nEpochs}: Acc=({r_acc:.2f}, {f_acc:.2f}, {af_acc:.2f}),\t\t")
                f.write(f"Cl2: Epoch {epoch + 1}/{args.nEpochs}: Acc=({r2_acc:.2f}, {f2_acc:.2f}, {af2_acc:.2f})\n\n")

if __name__ == '__main__':
    main()
    # r_acc, f_acc, af_acc = test()
    # print('[Test] [[Acc: %.2f, %.2f, %.2f]'% (r_acc, f_acc, af_acc))


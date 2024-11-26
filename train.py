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
from compose_models import RawNetWithFC, get_rawnet, get_ssdnet, get_wav2vec2_model
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
from utils import batch_audio_to_mel, batch_mel_to_audio
from visualize import compare_audio_samples
from tqdm import tqdm
import datetime
import yaml


# natsort_key = natsort_keygen(key = lambda y: y.lower())

parser = argparse.ArgumentParser()
parser.add_argument('--root', '-rt', type=str, default='../DATASETS/DTIM', help='')
parser.add_argument('--nEpochs', '-epoch', type=int, default=30, help='')
parser.add_argument('--batch_size', '-b', type=int, default=16, help='')
parser.add_argument('--num_workers', '-w', type=int, default=16, help='')
parser.add_argument('--lr', '-lr', type=float, default=0.0001, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0], help='')
args = parser.parse_args()
print(args)


time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
save_dir_path = str(time_now)
# Ensure the directory exists


# gpu_devices = ','.join([str(id) for id in args.gpu_devices])
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Device being used:', device)

G = Generator().to(device)
D = Discriminator().to(device)


# Function to transcribe audio to text using Wav2Vec 2.0
def transcribe_audio(audio, processor, model):

    input_values = processor(audio,sampling_rate=16000, return_tensors="pt").input_values

    input_values = input_values.to(device)
    input_values = input_values.squeeze(0)
    input_values = input_values.squeeze(1)
    input_values = input_values.half()  # Convert input to FP16

    # Get the predicted logits from the model
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids)
    return transcriptions


def get_transciption_loss(batch_text1, batch_text2):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(batch_text1) != len(batch_text2):
        raise ValueError("Both batches must have the same number of transcriptions.")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Pretrained semantic model

    # Encode both batches into embeddings
    embeddings1 = model.encode(batch_text1)
    embeddings2 = model.encode(batch_text2)
    
    # Calculate pairwise similarity and compute average loss
    losses = [
        1 - cosine_similarity([emb1], [emb2])[0][0]
        for emb1, emb2 in zip(embeddings1, embeddings2)
    ]
    return sum(losses) / len(losses)  # Average loss

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
t1_weight = 1
model_name = 'rawnet_3'
cl_model = get_rawnet(device)
cl_model2 = get_ssdnet(device)
t_processor_1, t_model_1 = get_wav2vec2_model(device)
save_dir_path = f'{save_dir_path}_{model_name}_{s1_w}_{s2_w}_{t1_weight}'

wav_dir_path = 'Wav_Plot_'+save_dir_path
os.makedirs(wav_dir_path, exist_ok=True)

checkpoint_dir_path = 'CHECKPOINTS_'+save_dir_path
os.makedirs(checkpoint_dir_path, exist_ok=True)

scaler = torch.GradScaler(device)

def sLoss(x, y, model):
    # logits = assist_model(x.squeeze(1))[1]  # Use the first item of the tuple
    # x = x
    # print(f"Input shape: {x.shape}")
    logits = model(x)  # x.squeeze(1) for aasist 
    # print(f"Logits: {str(logits[0])}  :   {str(logits[1])}")
    s_loss = classifiation_loss(logits, y.to(dtype=torch.long))
    # s_loss = classifiation_loss(assist_model(x.squeeze(1)), y.to(dtype=torch.long)) #+ classifiation_loss(inception(x), y.to(dtype=torch.long)) + \
             #classifiation_loss(mobilent(x), y.to(dtype=torch.long)) + classifiation_loss(resnet(x), y.to(dtype=torch.long)) + \
            #classifiation_loss(xception(x), y.to(dtype=torch.long))
    return s_loss

def train(epoch):
    g_losses = []
    c_losses = []
    d_losses = []
    t_losses = []

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

            forged_transciption = transcribe_audio(forged,t_processor_1,t_model_1)
            attacked_transciption = transcribe_audio(attacked,t_processor_1,t_model_1)
            if index == 0:
                forged_audio = forged[0].detach()  # Select first sample of forged audio
                attacked_audio = attacked[0].detach()   # Corresponding generated audio

                forged_t_samples = forged_transciption
                attacked_t_samples = attacked_transciption

                # Plot and compare
                compare_audio_samples(forged_audio, attacked_audio, forged_t_samples, attacked_t_samples,epoch,wav_dir_path, sr=16000)


            t_loss = get_transciption_loss(forged_transciption,attacked_transciption)

            per_loss = perceptual_loss(forged, attacked)
            adv_loss = adversarial_loss(y_real, D(attacked).squeeze().detach())
            c1_loss = sLoss(attacked.squeeze(1), y_real.to(dtype=torch.long),cl_model)
            c2_loss = sLoss(attacked, y_real.to(dtype=torch.long),cl_model2)

            g_loss = per_loss + adv_loss + s1_w * c1_loss+ s2_w*c2_loss + t1_weight*t_loss

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
        c_losses.append(c1_loss.item()+c2_loss.item())
        d_losses.append(d_loss.detach().cpu().numpy())
        t_losses.append(t_loss.item())

        # Update tqdm progress bar with current losses
        progress_bar.set_postfix({
            "G_Loss": f"{np.mean(g_losses):.4f}",
            "C_Loss": f"{np.mean(c_losses):.4f}",
            "D_Loss": f"{np.mean(d_losses):.4f}",
            "T_Loss": f"{np.mean(t_losses):.4f}",
        })

    progress_bar.close()  # Ensure tqdm closes cleanly when done
    return np.mean(g_losses), np.mean(c_losses), np.mean(d_losses), np.mean(t_losses)

def cal_acc(y, x, model):
    # outputs = inception(x)
    outputs = model(x)   # (x.squeeze(1))[1] for aasist   # ssdnet_model, assist_model 
    outputs = nn.Softmax(dim=-1)(outputs)
    _, y_ = torch.max(outputs, 1)

    acc = accuracy_score(y.cpu().numpy(), y_.cpu().numpy())

    return acc

def test(epoch=0):
    G.eval()
    real_acc, fake_acc, af_acc = [], [], []
    real2_acc,fake2_acc,af2_acc = [], [], []

    progress_bar = tqdm(test_loader, desc=f"Epoch {epoch}", unit="batch", leave=True)

    for test_sample in progress_bar:
        real = test_sample[0].unsqueeze(1).to(device, dtype=torch.float)
        forged = test_sample[2].unsqueeze(1).to(device, dtype=torch.float)

        y_real = torch.zeros(real.shape[0]).to(device, dtype=torch.float)
        y_fake = torch.ones(forged.shape[0]).to(device, dtype=torch.float)

        with torch.autocast(device_type='cuda', dtype=torch.float16):  # Enable Mixed Precision
            fake = G(forged)

            real_acc.append(cal_acc(y_real, real.squeeze(1),cl_model))
            fake_acc.append(cal_acc(y_fake, forged.squeeze(1),cl_model))
            af_acc.append(cal_acc(y_fake, fake.squeeze(1),cl_model))


            real2_acc.append(cal_acc(y_real, real,cl_model2))
            fake2_acc.append(cal_acc(y_fake, forged,cl_model2))
            af2_acc.append(cal_acc(y_fake, fake,cl_model2))
            
    progress_bar.close()
    return 100 * np.mean(real_acc), 100 * np.mean(fake_acc), 100 * np.mean(af_acc), 100 * np.mean(real2_acc), 100 * np.mean(fake2_acc), 100 * np.mean(af2_acc) 


def main():
    print('Training on', 2*len(train_dataset), 'and validating on ', 2*len(test_dataset), 'samples.')

    for epoch in range(args.nEpochs):
        g_loss, c_loss, d_loss, t_loss = train(epoch)

        scheduler_G.step()
        scheduler_D.step()

        print("[Train] [Epoch %d/%d], [LR: G=%f, D=%f], [C loss: %f], [D loss: %f], [G loss: %f], [T loss: %f]" % (
            epoch, args.nEpochs, scheduler_G.get_last_lr()[0],  scheduler_D.get_last_lr()[0], c_loss, d_loss, g_loss, t_loss))

        checkpoints = {
            'epoch': epoch+1,
            'state_dict': G.state_dict()
        }

        # Save the checkpoint
        torch.save(checkpoints, osp.join(checkpoint_dir_path, f'generator_{epoch+1}.pth'))

        r_acc, f_acc, af_acc, r2_acc, f2_acc, af2_acc = test(epoch)

        print('[Test Cl_1] [Epoch %d/%d], [Acc: %.2f, %.2f, %.2f]'% (epoch, args.nEpochs, r_acc, f_acc, af_acc))
        print('[Test Cl_2] [Epoch %d/%d], [Acc: %.2f, %.2f, %.2f]'% (epoch, args.nEpochs, r2_acc, f2_acc, af2_acc))
        # Save test accuracies to a file
        results_file = osp.join(checkpoint_dir_path, 'test_accuracies.txt')
        with open(results_file, 'a') as f:
            f.write(f"Cl1: Epoch {epoch + 1}/{args.nEpochs}: Acc=({r_acc:.2f}, {f_acc:.2f}, {af_acc:.2f})\n")
            f.write(f"Cl2: Epoch {epoch + 1}/{args.nEpochs}: Acc=({r2_acc:.2f}, {f2_acc:.2f}, {af2_acc:.2f})\n")

if __name__ == '__main__':
    main()
    # r_acc, f_acc, af_acc = test()
    # print('[Test] [[Acc: %.2f, %.2f, %.2f]'% (r_acc, f_acc, af_acc))


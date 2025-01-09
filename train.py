import os
import os.path as osp
import argparse
import numpy as np
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import StepLR
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from generator_simple import GeneratorSimple
from discriminator_simple import DiscriminatorSimple
from data_loader import DATAReader
from compose_models import get_ssdnet, get_inc_ssdnet, get_wav2vec2_model, get_speech_to_text_model
from utils import batch_audio_to_mel, batch_mel_to_audio, get_transciption_loss, transcribe_audio, transcribe_s2t
from visualize import compare_audio_samples
from tqdm import tqdm
import datetime
from torch.distributed import init_process_group, destroy_process_group

# torch.autograd.set_detect_anomaly(True)
# os.environ["RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["MASTER_ADDR"] = "127.0.0.1"
# os.environ["MASTER_PORT"] = "29500"

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--root', '-rt', type=str, default='../DATASETS/DTIM', help='Dataset root path')
parser.add_argument('--nEpochs', '-epoch', type=int, default=30, help='Number of epochs')
parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size')
parser.add_argument('--num_workers', '-w', type=int, default=2, help='Number of data loader workers')
parser.add_argument('--lr', '-lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--save_output', type=str, default="no", help='Save outputs or not')
parser.add_argument('--local_rank', type=int, default=0, help='Local rank for DDP')
args = parser.parse_args()
args.save_output = "no"

# # Initialize distributed processing
# torch.distributed.init_process_group(backend='nccl')
# local_rank = args.local_rank
# torch.cuda.set_device(local_rank)
# def ddp_setup():
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
init_process_group(backend="nccl")

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device('cuda', local_rank)

# # Logging setup
# if local_rank == 0:
#     print(f"Arguments: {args}")

# Define models
G = GeneratorSimple().to(device)
D = DiscriminatorSimple().to(device)
G = DDP(G, device_ids=[local_rank])
D = DDP(D, device_ids=[local_rank])

# Loss functions
perceptual_loss = nn.MSELoss()
adversarial_loss = nn.BCEWithLogitsLoss()
classification_loss = nn.CrossEntropyLoss()

# Optimizers and schedulers
optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999))
optimizer_D = torch.optim.SGD(D.parameters(), lr=args.lr)
scheduler_G = StepLR(optimizer_G, step_size=10, gamma=0.9)
scheduler_D = StepLR(optimizer_D, step_size=10, gamma=0.9)

# Load classification models
cl_model1 = DDP(get_ssdnet(device), device_ids=[local_rank])
cl_model2 = DDP(get_inc_ssdnet(device), device_ids=[local_rank])
t_processor_1, t_model_1 = get_wav2vec2_model(device)
t_processor_2, t_model_2 = get_speech_to_text_model(device)

model_name = "ssdnet_inc_ssdnet"
time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
save_dir_path = f"{time_now}_{model_name}"

# Loss weights
s1_w, s2_w, t1_w, t2_w = 0.0001, 0.0001, 0.8, 0.2
scaler = torch.GradScaler(device)

def sLoss(x, y, model):
    logits = model(x)
    return classification_loss(logits, y.to(dtype=torch.long))

def train(epoch):
    train_dataset = DATAReader(args=args, split='TRAIN')
    sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    # G.train()
    # D.train()

    g_losses, d_losses, c1_losses, c2_losses, t1_losses, t2_losses = [], [], [], [], [], []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=True)

    for index, train_sample in enumerate(progress_bar):
        real = train_sample[0].unsqueeze(1).to(device, dtype=torch.float)
        forged = train_sample[2].unsqueeze(1).to(device, dtype=torch.float)
        y_real = torch.zeros(real.shape[0]).to(device, dtype=torch.float)
        y_fake = torch.ones(forged.shape[0]).to(device, dtype=torch.float)

        # Train Generator
        optimizer_G.zero_grad()

        # with torch.autocast(device_type='cuda', dtype=torch.float16):  # Enable Mixed Precision
        attacked = G(forged)

        forged_transciption1 = transcribe_audio(forged,t_processor_1,t_model_1,device)
        attacked_transciption1 = transcribe_audio(attacked,t_processor_1,t_model_1,device)

        forged_transciption2 = transcribe_s2t(forged,t_processor_2,t_model_2,device)
        attacked_transciption2 = transcribe_s2t(attacked,t_processor_2,t_model_2,device)

        if index == 0 and args.save_output in "yes":
            forged_audio = forged[0].detach()  # Select first sample of forged audio
            attacked_audio = attacked[0].detach()   # Corresponding generated audio

            # Plot and compare
            wav_dir_path = 'Wav_Plot_'+save_dir_path
            os.makedirs(wav_dir_path, exist_ok=True)
            compare_audio_samples(forged_audio, attacked_audio, forged_transciption1, attacked_transciption1, forged_transciption2, attacked_transciption2,epoch,wav_dir_path, sr=16000)


        t1_loss = get_transciption_loss(forged_transciption1,attacked_transciption1)
        t2_loss = get_transciption_loss(forged_transciption2,attacked_transciption2)

        per_loss = perceptual_loss(forged, attacked)
        adv_loss = adversarial_loss(y_real, D(attacked).squeeze())
        c1_loss = sLoss(attacked, y_real.to(dtype=torch.long),cl_model1)
        c2_loss = sLoss(attacked, y_real.to(dtype=torch.long),cl_model2)
        # c3_loss = sLoss(attacked.squeeze(1), y_fake.to(dtype=torch.long),cl_model3)   # used fake label just to revert the label  for rawnet
        # c3_loss = np.float32(c3_loss.item())   # required for rawnet_2

        print(f"C losses: {c1_loss}, {c2_loss}, T1 loss: {t1_loss}, T2 loss: {t2_loss}")
        g_loss = per_loss + adv_loss + s1_w*c1_loss+ s2_w*c2_loss + t1_w*t1_loss + t2_w*t2_loss  # s2_w*c2_loss + 

        g_loss.backward()
        optimizer_G.step()
        # Scale the loss and backpropagate
        # scaler.scale(g_loss).backward()
        # scaler.step(optimizer_G)
        # scaler.update()

        # Train Discriminator
        optimizer_D.zero_grad()
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        real_loss = adversarial_loss(y_real, D(real).squeeze())
        fake_loss = adversarial_loss(y_fake, D(attacked).squeeze())
        d_loss = (real_loss + fake_loss) / 2

        # d_loss = d_loss.clone()  # To ensure it's not modified

        d_loss.backward()
        optimizer_D.step()
        # scaler.scale(d_loss).backward()
        # scaler.step(optimizer_D)
        # scaler.update()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        # d_losses.append(d_loss.detach().cpu().numpy())

        c1_losses.append(c1_loss.item())
        c2_losses.append(c2_loss.item())
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
    test_dataset = DATAReader(args=args, split='TEST')
    sampler = DistributedSampler(test_dataset)

    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
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

        if args.save_output in "yes":
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

if __name__ == "__main__":
    main()
    destroy_process_group()
    # r_acc, f_acc, af_acc = test()
    # print('[Test] [[Acc: %.2f, %.2f, %.2f]'% (r_acc, f_acc, af_acc))


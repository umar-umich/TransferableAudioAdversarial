import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import torchmetrics
import os
import os.path as osp

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
from itertools import islice
import datetime


# time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
# wav_dir_path = 'Wav_Plot_'+str(time_now)
# # Ensure the directory exists
# os.makedirs(wav_dir_path, exist_ok=True)


def plot_waveform_and_mel(audio1, audio2,sample_index,generator_id, index,wav_dir_path, sr=16000, title1="Real", title2="Generated"):
    """
    Plot waveforms and Mel spectrograms of two audio samples side by side.
    """
    # Plot waveforms
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(audio1)
    axs[0, 0].set_title(f"{title1} Waveform")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Amplitude")

    axs[0, 1].plot(audio2)
    axs[0, 1].set_title(f"{title2} Waveform")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Amplitude")

    # Mel spectrograms
    mel1 = librosa.feature.melspectrogram(y=audio1, sr=sr, n_mels=128, fmax=sr // 2)
    mel2 = librosa.feature.melspectrogram(y=audio2, sr=sr, n_mels=128, fmax=sr // 2)

    librosa.display.specshow(librosa.power_to_db(mel1, ref=np.max), sr=sr, ax=axs[1, 0], y_axis='mel', x_axis='time')
    axs[1, 0].set_title(f"{title1} Mel Spectrogram")

    librosa.display.specshow(librosa.power_to_db(mel2, ref=np.max), sr=sr, ax=axs[1, 1], y_axis='mel', x_axis='time')
    axs[1, 1].set_title(f"{title2} Mel Spectrogram")

    plt.tight_layout()
    # plt.show()
    
    # plt.savefig(osp.join(wav_dir_path, f'Fig_{epoch+1}_{index}.png'))
    plt.savefig(osp.join(wav_dir_path, f'Fig_{sample_index}_{generator_id}_{index}.png'))
    # Close the figure to free memory
    plt.close(fig)


def calculate_psnr(audio1, audio2):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two audio signals.
    """
    mse = np.mean((audio1 - audio2) ** 2)
    if mse == 0:  # Avoid divide by zero
        return float('inf')
    max_pixel = 1.0  # Assuming audio is normalized to [-1, 1]
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_ssim(audio1, audio2):
    """
    Calculate SSIM (Structural Similarity Index) between two audio signals.
    """
    # Normalize audio to [0, 1] range as SSIM expects positive inputs
    audio1 = (audio1 - np.min(audio1)) / (np.max(audio1) - np.min(audio1))
    audio2 = (audio2 - np.min(audio2)) / (np.max(audio2) - np.min(audio2))
    return ssim(audio1, audio2, data_range=1.0)


def compare_audio_samples(real_audio, fake_audio, forged_transciption1, attacked_transciption1,sample_index, generator_id, index, wav_dir_path, sr=16000):
    """
    Compare two audio samples using PSNR and SSIM, and plot their waveforms and spectrograms.
    """
    # Convert tensors to numpy if necessary
    if isinstance(real_audio, torch.Tensor):
        real_audio = real_audio.squeeze().cpu().numpy()
    if isinstance(fake_audio, torch.Tensor):
        fake_audio = fake_audio.squeeze().cpu().numpy()

    # Plot waveforms and Mel spectrograms
    plot_waveform_and_mel(real_audio, fake_audio,sample_index,generator_id, index,wav_dir_path, sr)

    # Calculate PSNR
    # psnr_value = calculate_psnr(real_audio, fake_audio)
    # print(f"PSNR: {psnr_value:.2f}")

    # # Calculate SSIM
    # ssim_value = calculate_ssim(real_audio, fake_audio)
    # print(f"SSIM: {ssim_value:.4f}")

    # Write results to a text file
    # results_file_path = os.path.join(wav_dir_path, f"metrics_epoch_{epoch+1}.txt")
    # os.makedirs(wav_dir_path, exist_ok=True)  # Ensure the directory exists
    # with open(results_file_path, "a") as file:
    #     file.write("\n\n")
    #     file.write(f"Epoch: {epoch+1}\n")
    #     file.write(f"PSNR: {psnr_value:.2f}\n")
    #     file.write(f"SSIM: {ssim_value:.4f}\n")
    #     file.write("\nForged and Attacked Transcriptions (T1):\n")
    #     file.write(f"{'Forged Transcription':<100} | {'Attacked Transcription':<100}\n")
    #     file.write("=" * 205 + "\n")

        # for forged, attacked in zip(forged_transciption1, attacked_transciption1):
        #     file.write(f"{forged:<100} | {attacked:<100}\n")

        # file.write("\nForged and Attacked Transcriptions (T2):\n")
        # file.write(f"{'Forged Transcription':<100} | {'Attacked Transcription':<100}\n")
        # file.write("=" * 205 + "\n")

        # for forged, attacked in zip(forged_transciption2, attacked_transciption2):
        #     file.write(f"{forged:<100} | {attacked:<100}\n")

    # print(f"Metrics saved to {results_file_path}")






# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', '-b', type=int, default=32, help='')
# parser.add_argument('--num_workers', '-w', type=int, default=16, help='')

# args = parser.parse_args()
# print(args)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# # Load the AASIST model
# with open("./models/aasist/AASIST.conf", "r") as f_json:
#     assist_config = json.loads(f_json.read())
# model_config = assist_config["model_config"]
# # model_config = config["model_config"]

# # print(f'ASSIST Conf: {str(model_config)}')
# assist_model = Model_ASSIST(model_config)
# assist_model.load_state_dict(torch.load("./weights/AASIST.pth", map_location=device))
# assist_model = assist_model.to(device)  # Move model to the appropriate device
# assist_model.eval()  # Set the model to evaluation mode

# def cal_acc(model_name, y, x):
#     # outputs = inception(x)
#     outputs = {}
#     if 'AASIST' in model_name:
#         outputs = assist_model(x.squeeze(1))
#     outputs = nn.Softmax(dim=-1)(outputs[1])
#     _, y_ = torch.max(outputs, 1)

#     acc = accuracy_score(y.cpu().numpy(), y_.cpu().numpy())

#     return acc


# def test(model_name):
#     test_dataset = DATAReader(args=args, split='TEST')
#     test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

#     # print('Device being used:', device)

#     # G = nn.DataParallel(Generator())
#     G = Generator()


#     checkpoint = torch.load("./CHECKPOINTS_2/generator_1.pth", map_location=device, weights_only=False)
#     state_dict = checkpoint['state_dict']

#     # Remove 'module.' prefix from keys
#     # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#     G.load_state_dict(state_dict)
#     G.to(device)  # Move model to the appropriate device

#     G.eval()
#     real_acc, fake_acc, af_acc = [], [], []
#     for test_sample in islice(test_loader, 10):
#         real = test_sample[0].to(device, dtype=torch.float)
#         forged = test_sample[2].unsqueeze(1).to(device, dtype=torch.float)

#         y_real =  torch.ones(real.shape[0]).to(device, dtype=torch.float)
#         y_fake =  torch.zeros(forged.shape[0]).to(device, dtype=torch.float)

#         fake = G(forged)
#         # Compare a real sample and a generated sample
#         real_audio = forged[0].detach()  # Select first sample of forged audio
#         fake_audio = fake[0].detach()   # Corresponding generated audio

#         # Plot and compare
#         compare_audio_samples(real_audio, fake_audio, sr=16000)
#         # break

#         real_acc.append(cal_acc(model_name, y_real, real))
#         fake_acc.append(cal_acc(model_name, y_fake, forged))

#         af_acc_ = cal_acc(model_name, y_fake, fake)
#         print(f'AF Acc: {af_acc_}')
#         af_acc.append(af_acc_)

#     return 100*np.mean(real_acc), 100*np.mean(fake_acc), 100*np.mean(af_acc)

if __name__ == '__main__':
    model_name = 'AASIST'  # RawNet3, AASIST
    # test(model_name)
    # r_acc, f_acc, af_acc = test(model_name)
    # print('[Test] [[Acc: %.2f, %.2f, %.2f]'% (r_acc, f_acc, af_acc))
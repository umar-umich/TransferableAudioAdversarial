import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import torchmetrics


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




def plot_waveform_and_mel(audio1, audio2, sr=16000, title1="Real", title2="Generated"):
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
    plt.show()


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


def compare_audio_samples(real_audio, fake_audio, sr=16000):
    """
    Compare two audio samples using PSNR and SSIM, and plot their waveforms and spectrograms.
    """
    # Convert tensors to numpy if necessary
    if isinstance(real_audio, torch.Tensor):
        real_audio = real_audio.squeeze().cpu().numpy()
    if isinstance(fake_audio, torch.Tensor):
        fake_audio = fake_audio.squeeze().cpu().numpy()

    # Plot waveforms and Mel spectrograms
    plot_waveform_and_mel(real_audio, fake_audio, sr)

    # Calculate PSNR
    psnr_value = calculate_psnr(real_audio, fake_audio)
    print(f"PSNR: {psnr_value:.2f}")

    # Calculate SSIM
    ssim_value = calculate_ssim(real_audio, fake_audio)
    print(f"SSIM: {ssim_value:.4f}")




parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-b', type=int, default=32, help='')
args = parser.parse_args()
print(args)

test_dataset = DATAReader(args=args, split='TEST')
test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Device being used:', device)

# G = nn.DataParallel(Generator())
G = Generator()
G.load_state_dict(torch.load("./CHECKPOINTS/generator_15.pth", map_location=device))
G.to(device)

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

def cal_acc(y, x):
    # outputs = inception(x)
    outputs = assist_model(x.squeeze(1))
    outputs = nn.Softmax(dim=-1)(outputs[1])
    _, y_ = torch.max(outputs, 1)

    acc = accuracy_score(y.cpu().numpy(), y_.cpu().numpy())

    return acc

# Example usage in test function
def test():
    G.eval()
    real_acc, fake_acc, af_acc = [], [], []

    for test_sample in islice(test_loader, 10):  # Process the first 10 batches
        real = test_sample[0].to(device, dtype=torch.float)
        forged = test_sample[2].unsqueeze(1).to(device, dtype=torch.float)

        y_real =  torch.ones(real.shape[0]).to(device, dtype=torch.float)
        y_fake =  torch.zeros(forged.shape[0]).to(device, dtype=torch.float)

        fake = G(forged)

        # Compare a real sample and a generated sample
        real_audio = forged[0].detach()  # Select first sample of forged audio
        fake_audio = fake[0].detach()   # Corresponding generated audio

        # Plot and compare
        compare_audio_samples(real_audio, fake_audio, sr=16000)

        # break  # Only process one batch for demonstratio

        real_acc.append(cal_acc(y_real, real))
        fake_acc.append(cal_acc(y_fake, forged))
        af_acc.append(cal_acc(y_fake, fake))

    return 100*np.mean(real_acc), 100*np.mean(fake_acc), 100*np.mean(af_acc)

if __name__ == '__main__':
    model_name = 'AASIST'
    test(model_name)

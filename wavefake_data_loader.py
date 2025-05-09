import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
from glob import glob
import csv


class WaveFakeDATAReader(data.Dataset):
    def __init__(self, split=None, labels=None):
        # self.data_root = data_root
        self.split = split
        
        if self.split in 'WaveFake':
            # dev_labels_file = '/data/Umar/A_Datasets/release_in_the_wild/meta.csv'
            # labels = get_in_the_wild_labels(dev_labels_file)
            self.data_root = '/data/Umar/A_Datasets/WaveFake/ljspeech_hifiGAN'
            hifigan_files = self.list_files(label=1)  # 1 for fake
            self.data_root = '/data/Umar/A_Datasets/WaveFake/ljspeech_melgan'
            melgan_files = self.list_files(label=1)  # 1 for fake
            self.data_root = '/data/Umar/A_Datasets/WaveFake/ljspeech_waveglow'
            waveglow_files = self.list_files(label=1)  # 1 for fake
            # self.data_root = '/data/Umar/A_Datasets/WaveFake/ljspeech_multi_band_melgan'
            # multi_band_melgan_files = self.list_files(label=1)  # 1 for fake
            # self.data_root = '/data/Umar/A_Datasets/WaveFake/ljspeech_parallel_wavegan'
            # parallel_wavegan_files = self.list_files(label=1)  # 1 for fake
            # self.data_root = '/data/Umar/A_Datasets/WaveFake/jsut_multi_band_melgan'
            # jsut_multi_files = self.list_files(label=1)  # 1 for fake
            # self.data_root = '/data/Umar/A_Datasets/WaveFake/jsut_parallel_wavegan'
            # jsut_parallel_files = self.list_files(label=1)  # 1 for fake
            # self.data_root = '/data/Umar/A_Datasets/WaveFake/ljspeech_melgan_large'
            # melgan_large_files = self.list_files(label=1)  # 1 for fake
            # self.data_root = '/data/Umar/A_Datasets/WaveFake/ljspeech_full_band_melgan'
            # full_band_melgan_files = self.list_files(label=1)  # 1 for fake
             
            
        
        self.fake_files = hifigan_files + melgan_files + waveglow_files  #+ jsut_multi_files + jsut_parallel_files + multi_band_melgan_files + parallel_wavegan_files + melgan_large_files + full_band_melgan_files



            

        self.labels = torch.zeros(len(self.fake_files), dtype=torch.int16)

        # # List real and fake files
        # self.real_files = self.list_files(label=0)  # 0 for real
        # self.fake_files = self.list_files(label=1)  # 1 for fake

        # print(f"{split}, Real files: {len(self.real_files)}, Fake files: {len(self.fake_files)}")

        self.n = len(self.fake_files)  # Balance dataset
        
        print(f"Balanced dataset size: {self.n}")


    def __len__(self):
        return self.n

    def __getitem__(self, index):
        # Get a sample from real and fake categories
        # real_audio = self.real_files[index]
        fake_audio = self.fake_files[index]

        # Preprocess the audio files
        # real_data = load_preprocess_AASIST(real_audio)
        fake_data = load_preprocess_AASIST(fake_audio)

        # # Get file ID (for label lookup)
        # # real_file_id = os.path.splitext(os.path.basename(real_audio))[0]
        # fake_file_id = os.path.splitext(os.path.basename(fake_audio))[0]

        # # Get the corresponding labels (real=0, fake=1)
        # # real_label = self.labels.get(real_file_id, 0)  # Default to real if not found
        # fake_label = self.labels.get(fake_file_id, 1)  # Default to fake if not found

        return  fake_data, self.labels[index], fake_data, self.labels[index]

    def list_files(self, label):
        file_list = []
        dataset_path = self.data_root  # Assuming split is 'TRAIN' or 'TEST'

        for file_name in os.listdir(dataset_path):
            if file_name.endswith('.flac') or file_name.endswith('.wav'):  # Audio files of interest
                file_path = os.path.join(dataset_path, file_name)

                # Check label (real=0, fake=1)
                # file_id = os.path.splitext(file_name)[0]
                # if self.labels.get(file_id) == label:
                file_list.append(file_path)
                # file_list.append(file_path)


        return file_list




def load_preprocess_AASIST(path, cut=96000):   # 96000, 64600
    from torch import Tensor
    import librosa


    # X, _ = sf.read(path)
    X, _ = librosa.load(path, sr=16000, mono=True)  # Record_1.mp3    Derek_orig_1.wav
    X_pad = pad(X, cut)
    x_inp = Tensor(X_pad)
    if len(x_inp.shape) != 1:
        if x_inp.shape[-1] != 1:
            x_inp = x_inp.mean(dim=-1, keepdim=False)
    return x_inp


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


if __name__ == '__main__':
    # dev_labels_file = '/data/Umar/A_Datasets/release_in_the_wild/meta.csv'
    # labels = get_in_the_wild_labels(dev_labels_file)
    print('abc')

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_root', '-rt', type=str, default='/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_eval/flac', help='Root directory for the dataset')
    # parser.add_argument('--split', '-sp', type=str, default='TRAIN', help='Split of the dataset (e.g., TRAIN, TEST)')
    # parser.add_argument('--labels_file', '-lf', type=str, default='/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', help='Path to the labels file')
    # parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0, 1], help='GPU devices to use')
    
    # args = parser.parse_args()

    # # Get the labels from the provided text file
    # test_labels = get_labels(args.labels_file)

    # # Initialize the dataset and dataloader
    # train_dataset = DATAReader(args=args, split=args.split, labels=test_labels)
    # train_loader = data.DataLoader(train_dataset, batch_size=24, shuffle=True)

    # print('Total train files: ', len(train_dataset))

    # # Example: Iterate through the dataset
    # for batch_idx, (real_data, real_label, fake_data, fake_label) in enumerate(train_loader):
    #     print(f"Batch {batch_idx+1}")
    #     print(f"Real data shape: {real_data.shape}, Real label: {real_label}")
    #     print(f"Fake data shape: {fake_data.shape}, Fake label: {fake_label}")
    #     break  # Only for demonstration purposes, break after the first batch

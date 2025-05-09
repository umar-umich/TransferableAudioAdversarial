import argparse
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
from glob import glob
import csv

# from data_loader import get_labels, load_preprocess_AASIST

def get_labels(labels_file):
    labels = {}
    with open(labels_file, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) >= 5:  # Assuming the line structure is consistent
                file_id = parts[1].strip()  # Extract the file_id
                label = parts[-1].strip()   # Extract the label (e.g., "spoof" or "bonafide")
                labels[file_id] = 1 if label == 'spoof' else 0  # 1 for fake, 0 for real
    return labels


def get_in_the_wild_labels(labels_file):
    labels = {}
    with open(labels_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',')  # Assuming tab-separated CSV, adjust delimiter if needed
        # print("CSV Headers:", reader.fieldnames)  # This will show you the headers of the CSV

        for row in reader:
            file_id = os.path.splitext(row['file'].strip())[0]  # Strip extension from 'file' (e.g., '0.wav' -> '0')
            label = row['label'].strip()   # Extract the label (e.g., 'spoof' or 'bona-fide')
            
            # Map the label to 1 (spoof) or 0 (bona-fide)
            labels[file_id] = 1 if label == 'spoof' else 0
    
    return labels


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


class RawNetDATAReader(data.Dataset):
    def __init__(self, args=None, split=None, labels=None):
        self.args = args
        if split == 'TRAIN':
            labels_file = '/data/Shared_Audio/A_Datasets/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt'
            labels = get_labels(labels_file)
            self.data_root = '/data/Shared_Audio/A_Datasets/ASV_2019/ASVspoof2019_LA_train/flac'
        elif split == 'TEST':
            labels_file = '/data/Shared_Audio/A_Datasets/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
            labels = get_labels(labels_file)
            self.data_root = '/data/Shared_Audio/A_Datasets/ASV_2019/ASVspoof2019_LA_eval/flac'
        elif split == 'DEV':
            labels_file = '/data/Shared_Audio/A_Datasets/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
            labels = get_labels(labels_file)
            self.data_root = '/data/Shared_Audio/A_Datasets/ASV_2019/ASVspoof2019_LA_dev/flac'
        elif split in 'In_The_Wild':
            dev_labels_file = '/data/Shared_Audio/A_Datasets/release_in_the_wild/meta.csv'
            labels = get_in_the_wild_labels(dev_labels_file)
            self.data_root = '/data/Shared_Audio/A_Datasets/release_in_the_wild'
        elif split in 'WaveFake':
            # dev_labels_file = '/data/Shared_Audio/A_Datasets/release_in_the_wild/meta.csv'
            # labels = get_in_the_wild_labels(dev_labels_file)
            self.data_root = '/data/Shared_Audio/A_Datasets/WaveFake/ljspeech_hifiGAN'

        self.labels = labels  # Provided labels for real/fake classification

        # Combine real and fake files into a single list with labels
        real_files_count = self.args.ratio*len(self.list_files(label=0))  # 0 for real
        
        # Initialize file paths list
        self.file_paths = []
        
        # Limit the number of files for each label to the minimum count
        for label in [0, 1]:  # 0 for real, 1 for fake
            files = self.list_files(label)
            # Only include the first 'min_files_count' files for each label
            self.file_paths.extend([(path, label) for path in files[:min(real_files_count,len(files))]])


        print(f"{split}, Total samples: {len(self.file_paths)}")

        # Set n to the minimum count of both labels to ensure balanced dataset
        self.n = len(self.file_paths) # real_files_count
        print(f'Balanced samples: {self.n}')
        # Count the number of real and fake samples
        real_count = sum(1 for _, label in self.file_paths if label == 0)
        fake_count = sum(1 for _, label in self.file_paths if label == 1)

        print(f"Real samples: {real_count}")
        print(f"Fake samples: {fake_count}")


    def __len__(self):
        # The dataset length is 2 * n (since you have both labels)
        return self.n

    def __getitem__(self, index):
        file_path, label = self.file_paths[index]

        # Load and preprocess the audio file
        audio_data = load_preprocess_AASIST(file_path, cut=96000)

        return audio_data, label

    def list_files(self, label):
        file_list = []
        dataset_path = self.data_root

        for file_name in os.listdir(dataset_path):
            if file_name.endswith('.flac') or file_name.endswith('.wav'):  # Audio files of interest
                file_path = osp.join(dataset_path, file_name)

                # Check label (real=0, fake=1)
                file_id = osp.splitext(file_name)[0]
                if self.labels.get(file_id) == label:
                    file_list.append(file_path)

        return file_list


# Main script
parser = argparse.ArgumentParser()
parser.add_argument('--root', '-rt', type=str, default='../DATASETS/DTIM', help='')
parser.add_argument('--nEpochs', '-epoch', type=int, default=30, help='')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='')
parser.add_argument('--num_workers', '-w', type=int, default=32, help='')
parser.add_argument('--lr', '-lr', type=float, default=0.001, help='')
# parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0], help='')
args = parser.parse_args()
print(args)
args.ratio = 4

train_dataset = RawNetDATAReader(args=args, split='TRAIN')

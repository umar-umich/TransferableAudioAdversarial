import argparse
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
from glob import glob

from data_loader import get_labels, load_preprocess_AASIST


class RawNetDATAReader(data.Dataset):
    def __init__(self, args=None, split=None, labels=None):
        self.args = args
        if split == 'TRAIN':
            labels_file = '/data/Umar/A_Datasets/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
            labels = get_labels(labels_file)
            self.data_root = '/data/Umar/A_Datasets/ASV_2019/ASVspoof2019_LA_train/flac'
        elif split == 'TEST':
            labels_file = '/data/Umar/A_Datasets/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
            labels = get_labels(labels_file)
            self.data_root = '/data/Umar/A_Datasets/ASV_2019/ASVspoof2019_LA_eval/flac'
        elif split == 'DEV':
            labels_file = '/data/Umar/A_Datasets/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
            labels = get_labels(labels_file)
            self.data_root = '/data/Umar/A_Datasets/ASV_2019/ASVspoof2019_LA_dev/flac'

        self.labels = labels  # Provided labels for real/fake classification

        # Combine real and fake files into a single list with labels
        real_files_count = 1*len(self.list_files(label=0))  # 0 for real
        
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
            if file_name.endswith('.flac'):  # Audio files of interest
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

train_dataset = RawNetDATAReader(args=args, split='TRAIN')

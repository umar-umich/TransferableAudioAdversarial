import torch.utils.data as data
import os
import os.path as osp
import soundfile as sf
import numpy as np
from glob import glob

class DATAReader(data.Dataset):
    def __init__(self, args=None, split=None, labels=None):
        self.args = args
        if split in 'TRAIN':
            train_labels_file = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
            labels = get_labels(train_labels_file)
            args.data_root = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_train/flac'
            # load corresponding labels
        elif split in 'TEST':
            test_labels_file = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
            labels = get_labels(test_labels_file)
            args.data_root = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_eval/flac'
        # self.split = split
        self.labels = labels  # Provided labels for real/fake classification

        # List real and fake files
        self.real_files = self.list_files(args, label=0)  # 0 for real
        self.fake_files = self.list_files(args, label=1)  # 1 for fake

        print(f"Real files: {len(self.real_files)}, Fake files: {len(self.fake_files)}")

        self.n = min(len(self.real_files), len(self.fake_files))  # Balance dataset

    def __len__(self):
        self.n = min(len(self.real_files), len(self.fake_files))  # Balance dataset

        return self.n

    def __getitem__(self, index):
        # Get a sample from real and fake categories
        real_audio = self.real_files[index]
        fake_audio = self.fake_files[index]

        # Preprocess the audio files
        real_data = load_preprocess_AASIST(real_audio)
        fake_data = load_preprocess_AASIST(fake_audio)

        # Get file ID (for label lookup)
        real_file_id = os.path.splitext(os.path.basename(real_audio))[0]
        fake_file_id = os.path.splitext(os.path.basename(fake_audio))[0]

        # Get the corresponding labels (real=0, fake=1)
        real_label = self.labels.get(real_file_id, 0)  # Default to real if not found
        fake_label = self.labels.get(fake_file_id, 1)  # Default to fake if not found

        return real_data, real_label, fake_data, fake_label

    def list_files(self, args, label):
        file_list = []
        dataset_path = self.args.data_root  # Assuming split is 'TRAIN' or 'TEST'

        for file_name in os.listdir(dataset_path):
            if file_name.endswith('.flac'):  # Audio files of interest
                file_path = os.path.join(dataset_path, file_name)

                # Check label (real=0, fake=1)
                file_id = os.path.splitext(file_name)[0]
                if self.labels.get(file_id) == label:
                    file_list.append(file_path)
                # file_list.append(file_path)


        return file_list


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


def load_preprocess_AASIST(path, cut=64600):
    from torch import Tensor
    import soundfile as sf

    X, _ = sf.read(path)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', '-rt', type=str, default='/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_eval/flac', help='Root directory for the dataset')
    parser.add_argument('--split', '-sp', type=str, default='TRAIN', help='Split of the dataset (e.g., TRAIN, TEST)')
    parser.add_argument('--labels_file', '-lf', type=str, default='/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', help='Path to the labels file')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0, 1], help='GPU devices to use')
    
    args = parser.parse_args()

    # Get the labels from the provided text file
    test_labels = get_labels(args.labels_file)

    # Initialize the dataset and dataloader
    train_dataset = DATAReader(args=args, split=args.split, labels=test_labels)
    train_loader = data.DataLoader(train_dataset, batch_size=24, shuffle=True)

    print('Total train files: ', len(train_dataset))

    # Example: Iterate through the dataset
    for batch_idx, (real_data, real_label, fake_data, fake_label) in enumerate(train_loader):
        print(f"Batch {batch_idx+1}")
        print(f"Real data shape: {real_data.shape}, Real label: {real_label}")
        print(f"Fake data shape: {fake_data.shape}, Fake label: {fake_label}")
        break  # Only for demonstration purposes, break after the first batch

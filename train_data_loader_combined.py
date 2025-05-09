import argparse
import random
import os
import os.path as osp
import numpy as np
from glob import glob
import csv
import librosa
from collections import Counter

# ... (Existing get_labels, get_in_the_wild_labels, pad, load_preprocess_AASIST functions) ...
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

class CombinedDATAReader(data.Dataset):
    def __init__(self, datasets_root, dataset_names, test_data=None, reuse_test=False):   # Add dataset_names
        self.datasets_root = datasets_root
        self.splits = ["train","eval","dev"]
        self.dataset_names = dataset_names  # Store dataset names
        self.file_paths = []
        self.labels = {}
        self.test_data = []  # Store test data within the class

        # For each dataset, read train and test files and corresponding labels file and add 80% to train_set and 20% to test set
        # assign train data and labels to current object and return the test files,labels
        # In next call, check if flag is on, just reuse the data and labels
        if reuse_test and test_data is not None:
            # Reuse test data if flag is set
            self.file_paths = test_data
        else:
            test_data = []
            for split in self.splits:
                for dataset_name in self.dataset_names:
                    data_root = os.path.join(self.datasets_root, dataset_name) # Construct paths dynamically based on the dataset name
                    self.labels = self.load_labels(data_root, dataset_name, split, utterance_level = True)

                    if dataset_name == 'release_in_the_wild':
                        audio_files = glob(os.path.join(data_root, "*.wav"))                        
                    elif dataset_name == 'Halftruth':
                        audio_files = glob(os.path.join(data_root, f"HAD/HAD_{split}/{split}/*.wav"))
                    elif dataset_name == 'ASV_2019':
                        audio_files = glob(os.path.join(data_root, f"ASVspoof2019_LA_{split}/flac/*.flac"))

                    # Now find the number of real and fake 
                    real_files, fake_files = [], []
                    # real_labels, fake_labels = {}, {}

                    for file_name in audio_files:
                        base_name = osp.splitext(osp.basename(file_name))[0]
                        label = self.labels.get(base_name, -1)
                        if label == 0:
                            real_files.append(file_name)
                            # real_labels[base_name] = 0
                        elif label == 1:
                            fake_files.append(file_name)
                            # fake_labels[base_name] = 1

                    # Balance the classes *before* splitting
                    # print(f"Dataset before splitting: {dataset_name}, split: {split}, Real: {len(real_files)}, Fake: {len(fake_files)}")
                    min_len = min(len(real_files), len(fake_files))
                    if dataset_name == 'release_in_the_wild':
                        real_files.sort()
                        fake_files.sort()
                        # sort these real and fake files 
                        real_files = real_files[:min_len]
                        fake_files = fake_files[:min_len]
                        balanced_count = int(min_len * 0.8)
     
                        self.file_paths.extend([(path, 0) for path in real_files[:balanced_count]])
                        self.file_paths.extend([(path, 1) for path in fake_files[:balanced_count]])

                        # test set
                        self.test_data.extend([(path, 0) for path in real_files[balanced_count:]])
                        self.test_data.extend([(path, 1) for path in fake_files[balanced_count:]])
                        print(f"Dataset after splitting: {dataset_name}, Train: {len(self.file_paths)}, Test: {len(self.test_data)}, Balanced count: {min_len}")
                    elif dataset_name == 'ASV_2019':
                        # don't move outside as we need sorting in in th wild
                        real_files = real_files[:min_len]
                        fake_files = fake_files[:min_len]
                        if split == "train" or split == "dev":
                            self.file_paths.extend([(path, 0) for path in real_files])
                            self.file_paths.extend([(path, 1) for path in fake_files])
                        elif split == "eval":
                            self.test_data.extend([(path, 0) for path in real_files])
                            self.test_data.extend([(path, 1) for path in fake_files])
                        print(f"Dataset after splitting: {dataset_name}, Train: {len(self.file_paths)}, Test: {len(self.test_data)}, Balanced count: {min_len}")
                    elif dataset_name == 'Halftruth':
                        real_files = real_files[:min_len]
                        fake_files = fake_files[:min_len]
                        if split == "train":
                            self.file_paths.extend([(path, 0) for path in real_files])
                            self.file_paths.extend([(path, 1) for path in fake_files])
                        elif split == "dev":
                            self.test_data.extend([(path, 0) for path in real_files])
                            self.test_data.extend([(path, 1) for path in fake_files])
                        print(f"Dataset after splitting: {dataset_name}, Train: {len(self.file_paths)}, Test: {len(self.test_data)}, Balanced count: {min_len}")



                print(f"{split}, Train samples: {len(self.file_paths)}, Test samples: {len(self.test_data)}")

        # print(f"{split}, Total samples: {len(self.file_paths)}")
        self.n = len(self.file_paths)
        print(f'Total samples: {self.n}')

    # Load segment-level or utterance-level labels
    def load_labels(self, data_root, dataset_name, part_name, utterance_level = True):
        labels = {}       
        if dataset_name == 'Halftruth':
            if utterance_level:
                utterance_label_path = os.path.join(data_root, f"HAD/HAD_{part_name}/HAD_{part_name}_label.txt")
                with open(utterance_label_path, 'r') as file:
                    for line in file:
                        parts = line.strip().split(' ')
                        if len(parts) >= 3:  # Ensure there are at least 3 parts
                            file_id = parts[0].strip()
                            label = parts[-1].strip()
                            labels[file_id] = int(label)  # Convert label to integer (0 or 1)
            # else:
            # TODO:
        elif dataset_name == 'ASV_2019':
            utterance_label_path = os.path.join(data_root, f"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{part_name}.trl.txt")               
            with open(utterance_label_path, 'r') as file:
                for line in file:
                    parts = line.strip().split(' ')
                    if len(parts) >= 3:
                        file_id = parts[1].strip()
                        label = parts[-1].strip()
                        labels[file_id] = 0 if label == 'spoof' else 1
        elif dataset_name == 'release_in_the_wild':
            labels_file = '/data/Shared_Audio/A_Datasets/release_in_the_wild/meta.csv'
            with open(labels_file, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter=',')  # Assuming tab-separated CSV, adjust delimiter if needed
                # print("CSV Headers:", reader.fieldnames)  # This will show you the headers of the CSV

                for row in reader:
                    file_id = os.path.splitext(row['file'].strip())[0]  # Strip extension from 'file' (e.g., '0.wav' -> '0')
                    label = row['label'].strip()   # Extract the label (e.g., 'spoof' or 'bona-fide')
                    
                    # Map the label to 1 (spoof) or 0 (bona-fide)
                    labels[file_id] = 1 if label == 'spoof' else 0
            
            return labels

        
        return labels
    # ... (rest of the class methods: __len__, __getitem__) ...
    def __len__(self):
        # The dataset length is 2 * n (since you have both labels)
        return self.n

    def __getitem__(self, index):
        file_path, label = self.file_paths[index]

        # Load and preprocess the audio file
        audio_data = load_preprocess_AASIST(file_path, cut=96000)

        return audio_data, label
        
    def get_test_dataset(self):
        """Returns a new dataset object for the test set."""
        return CombinedDATAReader(datasets_root=self.datasets_root, dataset_names=self.dataset_names, test_data=self.test_data, reuse_test=True)
    




if __name__ == "__main__":
    datasets_root = "/data/Shared_Audio/A_Datasets"   # Awais local PC: "/mnt/f/Awais_data/Datasets"
    dataset_names = ["release_in_the_wild", "ASV_2019", "Halftruth"]
    train_dataset = CombinedDATAReader(datasets_root, dataset_names)
    test_dataset = train_dataset.get_test_dataset()

    # ... Example usage ...
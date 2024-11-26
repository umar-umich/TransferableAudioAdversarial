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
            labels_file = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
            labels = get_labels(labels_file)
            args.data_root = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_train/flac'
        elif split == 'TEST':
            labels_file = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
            labels = get_labels(labels_file)
            args.data_root = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_eval/flac'
        elif split == 'DEV':
            labels_file = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
            labels = get_labels(labels_file)
            args.data_root = '/media/mufarooq/SSD_SMILES/Umar/UMFlint/Research/AA_Audio/ASV_2019/ASVspoof2019_LA_dev/flac'

        self.labels = labels  # Provided labels for real/fake classification

        # Combine real and fake files into a single list with labels
        self.file_paths = []
        for label in [0, 1]:  # 0 for real, 1 for fake
            self.file_paths.extend([(path, label) for path in self.list_files(args, label)])

        print(f"{split}, Total samples: {len(self.file_paths)}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path, label = self.file_paths[index]

        # Load and preprocess the audio file
        audio_data = load_preprocess_AASIST(file_path)

        return audio_data, label

    def list_files(self, args, label):
        file_list = []
        dataset_path = self.args.data_root

        for file_name in os.listdir(dataset_path):
            if file_name.endswith('.flac'):  # Audio files of interest
                file_path = osp.join(dataset_path, file_name)

                # Check label (real=0, fake=1)
                file_id = osp.splitext(file_name)[0]
                if self.labels.get(file_id) == label:
                    file_list.append(file_path)

        return file_list

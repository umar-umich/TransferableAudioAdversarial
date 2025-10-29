import torch.utils.data as data
import os
import os.path as osp
import numpy as np
from glob import glob
import csv


class DF_DATAReader(data.Dataset):
    def __init__(self, file_path=None):
        # self.data_root = data_root
        self.file_path = file_path
        
        self.audio_data = load_preprocess_AASIST(file_path)



        self.n = len(self.audio_data)  # Balance dataset
        
        print(f"Data size: {self.n}")


    def __len__(self):
        return self.n

    def __getitem__(self, index):
        # Get a sample from real and fake categories
        audio_data = self.audio_data[index]

        return audio_data
    
    def list_files(self, label):
        file_list = []
        dataset_path = self.data_root  # Assuming split is 'TRAIN' or 'TEST'

        for file_name in os.listdir(dataset_path):
            if file_name.endswith('.flac') or file_name.endswith('.wav'):  # Audio files of interest
                file_path = os.path.join(dataset_path, file_name)

                # Check label (real=0, fake=1)
                file_id = os.path.splitext(file_name)[0]
                if self.labels.get(file_id) == label:
                    file_list.append(file_path)
                # file_list.append(file_path)


        return file_list



def load_preprocess_AASIST(path, chunk_size=64000 ):  # 64000, 96000
    from torch import Tensor
    import librosa
    import numpy as np

    # Load audio file
    X, _ = librosa.load(path, sr=16000, mono=True)

    # Divide audio into chunks
    X_chunks = chunk_audio(X, chunk_size)

    # Convert to Tensor
    x_inp = Tensor(X_chunks)

    # Handle shape adjustment if necessary
    if len(x_inp.shape) != 2:
        raise ValueError("Expected 2D Tensor with shape (num_chunks, chunk_size). Check your input audio.")

    return x_inp

def chunk_audio(x, chunk_size):
    """
    Divide the audio into multiple chunks of the given chunk size.
    If the audio length is not a multiple of chunk_size, pad the last chunk with zeros.
    
    Args:
        x (numpy.ndarray): Input audio signal.
        chunk_size (int): Desired chunk size.

    Returns:
        numpy.ndarray: 2D array with shape (num_chunks, chunk_size).
    """
    x_len = len(x)
    num_chunks = int(np.ceil(x_len / chunk_size))

    # Pad the audio to make its length a multiple of chunk_size
    padded_len = num_chunks * chunk_size
    padded_x = np.pad(x, (0, padded_len - x_len), mode='constant')

    # Reshape into chunks
    chunks = padded_x.reshape(num_chunks, chunk_size)
    return chunks


if __name__ == '__main__':
    test_audio_path = './Ulendo_test_sample.wav'  # Example path to audio file
    data_reader = DATAReader(test_audio_path)

    print(f"Number of chunks: {len(data_reader)}")

    # Access individual chunks
    for i in range(len(data_reader)):
        chunk = data_reader[i]
        print(f"Chunk {i+1}: Shape {chunk.shape}")

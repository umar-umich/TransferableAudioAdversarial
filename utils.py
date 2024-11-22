import torch
import librosa
import numpy as np

def batch_audio_to_mel(batch_audio, sr, n_mels=128, n_fft=2048, hop_length=512):
    """
    Convert a batch of raw audio signals to Mel spectrograms.
    Args:
        batch_audio: Tensor of shape (batch_size, 1, audio_length).
        sr: Sample rate.
        n_mels: Number of Mel bands.
        n_fft: Length of the FFT window.
        hop_length: Number of samples between successive frames.
    Returns:
        Mel spectrograms as a tensor of shape (batch_size, 1, n_mels, time_steps).
    """
    mel_list = []
    for audio in batch_audio.squeeze(1).cpu().numpy():  # Remove channel dimension and convert to numpy
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmax=sr // 2
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_list.append(mel_db)
    # Convert list of NumPy arrays to a single NumPy array
    mel_array = np.stack(mel_list, axis=0)  # Shape: (batch_size, n_mels, time_steps)

    
    mel_tensor = torch.tensor(mel_array).unsqueeze(1).float()  # Add batch and channel dimensions
    return mel_tensor.to(batch_audio.device)

def batch_mel_to_audio(batch_mel, sr, n_fft=2048, hop_length=512, original_length=64600):
    """
    Convert a batch of Mel spectrograms back to raw audio signals.
    Args:
        batch_mel: Tensor of shape (batch_size, 1, n_mels, time_steps).
        sr: Sample rate.
        n_fft: Length of the FFT window.
        hop_length: Number of samples between successive frames.
        original_length: Desired output length for audio.
    Returns:
        Raw audio signals as a tensor of shape (batch_size, 1, audio_length).
    """
    audios = []
    for mel in batch_mel.squeeze(1).detach().cpu().numpy():
        audio = librosa.feature.inverse.mel_to_audio(mel, sr=sr, n_fft=n_fft, hop_length=hop_length)
        audio = librosa.util.fix_length(audio, size=original_length)  # Ensure consistent length
        audios.append(audio)
    audio_array = np.stack(audios, axis=0)  # Shape: (batch_size, audio_length)

    audio_tensor = torch.tensor(audio_array).unsqueeze(1).float()  # Add channel dimension
    return audio_tensor.to(batch_mel.device)


# def batch_mel_to_audio(batch_mel, sr):
#     """
#     Convert a batch of Mel spectrograms back to raw audio.
#     Args:
#         batch_mel: Tensor of shape (batch_size, 1, n_mels, time_steps).
#         sr: Sample rate.
#     Returns:
#         Reconstructed audio signals as a tensor of shape (batch_size, 1, audio_length).
#     """
#     audio_list = []
#     for mel in batch_mel.squeeze(1).detach().cpu().numpy():  # Remove channel dimension and convert to numpy
#         mel_power = librosa.db_to_power(mel)  # Convert decibel to power spectrogram
#         audio = librosa.feature.inverse.mel_to_audio(mel_power, sr=sr)
#         audio_list.append(audio)
    
#     max_length = max(len(a) for a in audio_list)
#     padded_audio = [np.pad(a, (0, max_length - len(a))) for a in audio_list]  # Pad audio to the same length
#     audio_tensor = torch.tensor(padded_audio).unsqueeze(1).float()  # Add batch and channel dimensions
#     return audio_tensor.to(batch_mel.device)

if __name__ == "__main__":
    # Sample input
    batch_audio = torch.randn(32, 1, 64600).to('cuda')  # Batch of audio signals
    sr = 16000

    # Convert to Mel spectrograms
    mel_specs = batch_audio_to_mel(batch_audio, sr)

    # Pass Mel spectrograms through your generator model (example)
    # generated_mel_specs = G(mel_specs)

    # Convert back to raw audio
    generated_audio = batch_mel_to_audio(mel_specs, sr)

import torch
import librosa
import numpy as np

from compose_models import get_speech_to_text_model

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


# Function to transcribe audio to text using Wav2Vec 2.0
def transcribe_audio(audio, processor, model,device):

    input_values = processor(audio,sampling_rate=16000, return_tensors="pt").input_values

    input_values = input_values.to(device)
    input_values = input_values.squeeze(0)
    input_values = input_values.squeeze(1)
    input_values = input_values.half()  # Convert input to FP16

    # Get the predicted logits from the model
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the predicted logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids)
    return transcriptions


# Function to transcribe audio to text using Wav2Vec 2.0
def transcribe_s2t(audio, processor, model,device):

    audio = audio.squeeze(1).detach().cpu().numpy()
    input_features = processor(
        audio,
        sampling_rate=16_000,
        return_tensors="pt"
    ).input_features

    # Ensure input features match the model's dtype and are on the correct device
    input_features = input_features.to(device).to(model.dtype)

    # Generate transcription
    generated_ids = model.generate(input_features=input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return transcription


def get_transciption_loss(batch_text1, batch_text2):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(batch_text1) != len(batch_text2):
        raise ValueError("Both batches must have the same number of transcriptions.")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Pretrained semantic model

    # Encode both batches into embeddings
    embeddings1 = model.encode(batch_text1)
    embeddings2 = model.encode(batch_text2)
    
    # Calculate pairwise similarity and compute average loss
    losses = [
        1 - cosine_similarity([emb1], [emb2])[0][0]
        for emb1, emb2 in zip(embeddings1, embeddings2)
    ]
    return sum(losses) / len(losses)  # Average loss

if __name__ == "__main__":
    # Sample input
    # batch_audio = torch.randn(32, 1, 96000)#.to('cuda')  # Batch of audio signals
    path = "./Derek_orig_1.wav"
    X, _ = librosa.load(path, sr=16000, mono=True)  # Record_1.mp3    Derek_orig_1.wav

    sr = 16000
    device='cuda'
    processor, model = get_speech_to_text_model(device)

    forged_transciption = transcribe_s2t(X,processor, model,device)
    print(str(forged_transciption))


import librosa
import numpy as np

def load_audio(path, data_params):
  audio, _ = librosa.load(path, sr=data_params['fs'], mono=data_params['mono'])

  audio = audio.T
  audio = np.reshape(audio, [-1, 1])
  return audio

def normalize_audio(audio):
  mean = np.mean(audio)
  audio -= mean
  max = max(abs(audio))

  return audio / max

import librosa
import numpy as np

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    tempo = librosa.beat.tempo(y=y, sr=sr)
    energy = librosa.feature.rms(y=y)

    return {
        'mfcc_mean': np.mean(mfcc, axis=1),
        'pitch_mean': np.mean(pitch),
        'tempo': tempo[0],
        'energy_mean': np.mean(energy)
    }

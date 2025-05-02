import librosa
import numpy as np

# audio_path is a place holder for now
# remove " " and input actual path.
y, sr = librosa.load("audio_path") 
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
pitch = librosa.yin(y, fmin=50, fmax=300)
tempo = librosa.beat.tempo(y=y, sr=sr)
energy = librosa.feature.rms(y=y)

features = {
    'mfcc_mean': np.mean(mfcc, axis=1),
    'pitch_mean': np.mean(pitch),
    'tempo': tempo[0],
    'energy_mean': np.mean(energy)
}

from auth import authenticate_huggingface
from datasets import load_dataset
from extract import extract_features

authenticate_huggingface()

# Load the dataset
dataset = load_dataset("google/speech_commands", split="train")

# Access a sample
sample = dataset[0]
audio = sample['audio']

# Extract features
waveform = audio['array']
sampling_rate = audio['sampling_rate']
features = extract_features(waveform, sampling_rate)

print(features)


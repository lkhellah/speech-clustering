from datasets import load_dataset
from auth import authenticate_huggingface
from extract import extract_features
from itertools import islice
import pandas as pd
import json


"""
Script to:
1. Authenticate with Hugging Face using .env token
2. Stream audio samples from the Google Speech Commands dataset
3. Filter for samples with utterance:'backward' only (temporary, will be scaled up)
4. Ensure one sample per unique speaker
5. Extract acoustic features (MFCCs, pitch, tempo, energy)
6. Build and display a pandas DataFrame
"""

# Step 1: Authenticate
authenticate_huggingface()

# Step 2: Stream dataset
streamed_dataset = load_dataset(
    "google/speech_commands",
    "v0.02",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

# Step 3: Filter for 'backward' samples only
label_names = streamed_dataset.features["label"].names
target_label_id = label_names.index("backward")

seen_speakers = set()
filtered_samples = []

for sample in streamed_dataset:
    if sample["label"] == target_label_id and sample["speaker_id"] not in seen_speakers:
        seen_speakers.add(sample["speaker_id"])
        filtered_samples.append(sample)
    if len(filtered_samples) == 200:  # limit for quick experimentation
        break

# Step 4: Extract features
data = []
for sample in filtered_samples:
    audio = sample["audio"]
    waveform = audio["array"]
    sr = audio["sampling_rate"]
    features = extract_features(waveform, sr)
    features["speaker_id"] = sample["speaker_id"]
    data.append(features)

# Step 5: Create and inspect DataFrame
df = pd.DataFrame(data)
print(df.head())
# this line allows me to save the entire DataFrame to a .csv file
# because plain lists or NumPy arrays aren’t natively supported in CSVs.
df["mfcc_mean"] = df["mfcc_mean"].apply(lambda x: json.dumps(x.tolist()))
df.to_csv("backward_features.csv", index=False)

# Now to cluster, have to convert from string back to lists



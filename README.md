# Clustering Spoken Audio Samples by Delivery Style using Acoustic Features

Portland State University – Data Clustering Course Final Project
This project clusters 200 spoken samples of the word **“backward”** from the Google Speech Commands dataset. Using acoustic features like MFCCs, pitch, tempo, and energy, the goal is to group samples by **how** the word is spoken - focusing on delivery style rather than content. The word is kept constant to isolate stylistic differences in speech. 

# Project Setup

Note:
This project uses a Hugging Face token to authenticate and stream data from the Google Speech Commands dataset. To reproduce the code, you'll need to create a Hugging Face account and store your token in a .env file as described in the auth.py script.


### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the dataset loading script

```bash
python load_dataset.py
```

This script

- Authenticates with Hugging Face Hub using a token stored in your .env file
- Streams the Google Speech Commands dataset
- Filters and loads 200 samples labeled "backward"
- Extracts acoustic features for each audio sample:
  - MFCCs (Mel-frequency cepstral coefficients)
  - Pitch
  - Tempo
  - Energy
- Saves the extracted features to backward_features.csv for clustering and analysis in the notebook

### Step 3:
Open and run the cells in `demo-notebook.ipynb` to:
- Perform KMeans, DBSCAN, and Agglomerative clustering
- Analyze cluster patterns
- Visualize results
- Play example audio clips from each cluster
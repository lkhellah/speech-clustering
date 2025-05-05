# Project Setup

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the dataset loading script
```bash
python load_dataset.py
```

This script:
- Authenticates with Hugging Face using  .env file
- Loads the Google Speech Commands dataset
- Extracts and prints features from a single audio sample (NOW for testing purposes, changing later)



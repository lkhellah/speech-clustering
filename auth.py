# Log in to Hugging Face
import os
from dotenv import load_dotenv
from huggingface_hub import login


def authenticate_huggingface():

    # Load environment variables from .env file
    load_dotenv()

    # Get token after loading .env
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        raise ValueError("Hugging Face token not found in environment variables.")

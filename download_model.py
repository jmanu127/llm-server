import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-E4B-it")
HF_TOKEN = os.getenv("HF_TOKEN", "")

def main():
    if not HF_TOKEN or HF_TOKEN.strip() == "" or HF_TOKEN == "ADD_TOKEN_HERE":
        print("=" * 70)
        print("ERROR: Hugging Face Access Token (HF_TOKEN) is missing or set to placeholder.")
        print("Please follow these steps to configure your environment:")
        print("1. Create a Hugging Face account at https://huggingface.co")
        print("2. Accept the model license terms at https://huggingface.co/google/gemma-4-E4B-it")
        print("3. Generate a User Access Token (read) at https://huggingface.co/settings/tokens")
        print("4. Edit the '.env' file in this directory and replace 'ADD_TOKEN_HERE' with your token.")
        print("=" * 70)
        sys.exit(1)

    print(f"Attempting to pre-download model '{MODEL_ID}'...")
    print("This will download the model weights (~9GB) and cache them locally.")
    print("Please ensure you have a stable and fast internet connection.")
    print("-" * 50)

    try:
        from huggingface_hub import snapshot_download
        
        # Download tokenizer and model files
        snapshot_download(
            repo_id=MODEL_ID,
            token=HF_TOKEN,
            # We only want safetensors / configuration files, ignoring other legacy weights if present
            ignore_patterns=["*.gguf", "*.bin", "*.pth"]
        )
        print("-" * 50)
        print(f"SUCCESS: Model '{MODEL_ID}' has been cached successfully!")
        print("Ready for local or containerized execution.")
    except Exception as e:
        print("-" * 50)
        print(f"ERROR downloading model: {e}")
        print("If you get a 401 or 403 error, please double check that:")
        print("1. Your HF_TOKEN is correct and has 'read' access.")
        print("2. You have explicitly accepted the Gemma 4 license terms on Hugging Face.")
        sys.exit(1)

if __name__ == "__main__":
    main()

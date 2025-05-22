from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

def download_model():
    print("Starting NLLB model download...")
    model_name = "facebook/nllb-200-distilled-600M"
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    print(f"Downloading tokenizer to {cache_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    print(f"Downloading model to {cache_dir}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    
    print("Download complete!")
    print(f"Model files are stored in: {cache_dir}")

if __name__ == "__main__":
    download_model() 
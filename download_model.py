from pathlib import Path
import urllib.request

MODEL_DIR = Path("models/Xenova/all-MiniLM-L6-v2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/"
FILES = {
    "model.onnx": BASE_URL + "onnx/model.onnx",
    "tokenizer.json": BASE_URL + "tokenizer.json",
    "tokenizer_config.json": BASE_URL + "tokenizer_config.json",
}

print("Downloading ONNX model and tokenizer files...")
for filename, url in FILES.items():
    file_path = MODEL_DIR / filename
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, file_path)

print("Download complete!")
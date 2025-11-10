# download_model.py
import os

# Install gdown if not installed
try:
    import gdown
except ImportError:
    os.system('pip install gdown')
    import gdown

# Google Drive file ID
file_id = "1hoa_Ohg3VRJIaOIPPd4DBn8LXRr5Vs1i"

# Destination path
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "best.onnx")

# Google Drive URL
url = f"https://drive.google.com/uc?id={file_id}"

print("⬇️  Downloading model file from Google Drive...")
gdown.download(url, output_path, quiet=False)
print(f"✅ Model saved at: {output_path}")

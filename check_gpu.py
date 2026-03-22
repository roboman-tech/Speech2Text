"""
Check GPU and PyTorch CUDA setup.
Run: python check_gpu.py
"""
import subprocess
import sys

print("=== GPU / CUDA check ===\n")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    cuda_build = "+cu" in torch.__version__ or "cu1" in torch.__version__
    print(f"CUDA in build:   {cuda_build}")
    print(f"torch.cuda:      {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
    else:
        if not cuda_build:
            print("\n>>> PyTorch is CPU-only. To use GPU, reinstall with CUDA:")
            print("    pip uninstall torch -y")
            print("    pip install torch --index-url https://download.pytorch.org/whl/cu124")
            print("    (cu124 for CUDA 12.x, cu121 for older)")
except ImportError:
    print("PyTorch not installed.")

print("\n--- nvidia-smi ---")
try:
    subprocess.run(["nvidia-smi"], check=False)
except FileNotFoundError:
    print("nvidia-smi not found (no NVIDIA GPU or drivers)")

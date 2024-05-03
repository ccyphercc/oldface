import subprocess
import os
import torch

if torch.cuda.is_available():
  device="cuda"
  print("Using GPU")
else:
  device="cpu"
  print("Using CPU")

# Clone the repository
subprocess.run(["git", "clone", "https://github.com/facefusion/facefusion", "--branch", "2.5.2", "--single-branch"], check=True)
# chande directory to face fusion to run ui
os.chdir("facefusion")

# installation
subprocess.run(["python", "install.py", "--onnxruntime", "cuda-11.8", "--skip-conda"], check=True)

# Run the ui
if device=="cuda":
    subprocess.run(["python", "run.py", "--execution-providers", "cuda"], check=True)
else:
    subprocess.run(["python", "run.py", "--execution-providers", "cpu"], check=True)
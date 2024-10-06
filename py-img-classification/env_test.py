import platform
import torch

print(f"python: {platform.python_version()}")
print(f"torch {torch.__version__} CUDA: {torch.cuda.is_available()}")

x = torch.rand(5, 3)
print(x)

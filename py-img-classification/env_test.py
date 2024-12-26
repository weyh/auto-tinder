import platform

from commonlib.training_common import *


print(f"python: {platform.python_version()}")
print(f"torch {torch.__version__} device: {get_device()}")

x = torch.rand(5, 3)
print(x)

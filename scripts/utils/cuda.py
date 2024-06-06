import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# os.environ["WORLD_SIZE"] = "1"

import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

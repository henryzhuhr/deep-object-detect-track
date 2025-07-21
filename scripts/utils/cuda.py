import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ["WORLD_SIZE"] = "1"

import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

"""
# 设置所有 GPU 可见（编号从 0 开始映射）
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd,)

# 运行 Python 脚本
python -c "
import torch;
print(f'PyTorch sees {torch.cuda.device_count()} GPUs');
print(f'Current device index: {torch.cuda.current_device()}');
print(f'Device name: {torch.cuda.get_device_name(0)}');
"
"""

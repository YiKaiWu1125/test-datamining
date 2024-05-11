import torch

# 列出所有可用的 GPU
if torch.cuda.is_available():
    print("Available GPUs:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 当前激活的 GPU
print(f"Current GPU: {torch.cuda.current_device()}, {torch.cuda.get_device_name()}")
torch.cuda.set_device(0)
print(f"Current GPU after setting: {torch.cuda.current_device()}, {torch.cuda.get_device_name()}")
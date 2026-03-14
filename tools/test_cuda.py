try:
    import torch as _torch
    print("CUDA available:", _torch.cuda.is_available())
    print("CUDA devices:", _torch.cuda.device_count())
    for i in range(_torch.cuda.device_count()):
        print(i, _torch.cuda.get_device_name(i))
except Exception:
    print("CUDA not available:")
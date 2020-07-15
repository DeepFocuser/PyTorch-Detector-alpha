import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Allocated:', round(torch.cuda.memory_allocated(device)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(device)/1024**3,1), 'GB')

    total_memory = torch.cuda.get_device_properties(device).total_memory
    c = torch.cuda.memory_cached(device)
    a = torch.cuda.memory_allocated(device)
    free_memory = c - a  # free inside cache
    free_memory = round(free_memory / (1024 ** 3), 2)
    total_memory = round(total_memory / (1024 ** 3), 2)
    print(f'Running on {device} / free memory : {free_memory}GB / total memory {total_memory}GB')
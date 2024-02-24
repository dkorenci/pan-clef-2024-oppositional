'''
From: https://huggingface.co/docs/transformers/perf_train_gpu_one
'''

from pynvml import *
import torch

def print_gpu_utilization(tag='', device=0):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied{(' @'+tag+' ') if tag else ''}: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def print_cuda_devices():
    nvmlInit()
    if torch.cuda.is_available():
        print("CUDA Devices:")
        # Get original device indices based on CUDA_VISIBLE_DEVICES
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible_devices:
            device_indices = [int(ix) for ix in cuda_visible_devices.split(',')]
        else:
            device_indices = list(range(torch.cuda.device_count()))
        # print device data
        for i, original_index in enumerate(device_indices):
            props = torch.cuda.get_device_properties(i)
            handle = nvmlDeviceGetHandleByIndex(original_index)
            info = nvmlDeviceGetMemoryInfo(handle)
            print(f"Device {i} (system-level index: {original_index}): {torch.cuda.get_device_name(i)}")
            print(f"    mem.total: {props.total_memory/1e9:.2f} GB, mem.used: {info.used/1e9:.2f} GB")  # Memory in GB
    else:
        print("No CUDA devices available")

if __name__ == '__main__':
    print_cuda_devices()

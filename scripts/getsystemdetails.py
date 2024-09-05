import platform
import psutil
import os

# Try to import GPU-related module; if unavailable, skip GPU detection
try:
    import GPUtil
    gpu_available = True
except ImportError:
    gpu_available = False

def get_system_info():
    info = {}
    
    # Get platform information
    info['Platform'] = platform.system()
    info['Platform Version'] = platform.version()
    info['Architecture'] = platform.machine()
    info['Processor'] = platform.processor()
    info['Python Version'] = platform.python_version()

    # Get CPU information
    info['CPU Cores (Logical)'] = psutil.cpu_count(logical=True)
    info['CPU Cores (Physical)'] = psutil.cpu_count(logical=False)

    # Get Memory Information
    virtual_memory = psutil.virtual_memory()
    info['Total Memory (GB)'] = virtual_memory.total / (1024**3)
    
    # Get Disk Information
    disk_info = psutil.disk_usage('/')
    info['Total Disk Space (GB)'] = disk_info.total / (1024**3)

    # Get GPU information if available
    if gpu_available:
        gpus = GPUtil.getGPUs()
        if gpus:
            for idx, gpu in enumerate(gpus):
                info[f'GPU {idx} Name'] = gpu.name
                info[f'GPU {idx} Memory Total (GB)'] = gpu.memoryTotal / 1024
                info[f'GPU {idx} Driver Version'] = gpu.driver
        else:
            info['GPU'] = 'No GPU detected'
    else:
        info['GPU'] = 'GPUtil not installed or no GPU detected'

    return info

def print_system_info(info):
    print("System Configuration Overview:")
    for key, value in info.items():
        print(f"- {key}: {value}")

if __name__ == "__main__":
    system_info = get_system_info()
    print_system_info(system_info)

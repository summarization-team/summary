import torch
from transformers import pipeline


def get_available_devices():
    devices = []
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            devices.append(f'cuda:{i}')
    devices.append('cpu')
    return devices


def get_device():
    """
    Checks for GPU availability and returns the appropriate device ('cuda' or 'cpu').

    Returns:
        str: The device type. 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        # If a GPU is available, return 'cuda' to use it
        return 0
    else:
        # If no GPU is available, default to using the CPU
        return -1


# with open('gpu.test', 'w', encoding='utf-8') as outfile:
#     outfile.write(f"Job Initiated")

device_list = get_available_devices()
device = get_device()

# Create the pipeline
sentiment_pipeline = pipeline(task="sentiment-analysis", device=device)

# Use the pipeline for inference
results = sentiment_pipeline("I love using Hugging Face Transformers!")

with open('gpu.test', 'w', encoding='utf-8') as outfile:
    outfile.write(f"device list: {device_list}\n")
    outfile.write(f"device name: {device}\n")
    outfile.write(f"res:{results}\n")
    outfile.write("made it to end")

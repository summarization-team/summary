import torch
from transformers import pipeline


def get_available_devices():
    devices = []
    if torch.cuda.is_available():
        devices.append('cuda')
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            devices.append(f'cuda:{i}')
    devices.append('cpu')
    return devices

device_list = get_available_devices()

# Create the pipeline
sentiment_pipeline = pipeline("sentiment-analysis", device=0)

# Use the pipeline for inference
results = sentiment_pipeline("I love using Hugging Face Transformers!")


with open('gpu.test', 'w', encoding='utf-8') as outfile:
    outfile.write(f"device: {device_list}")
    outfile.writelines(results)

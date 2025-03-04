from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file
import torch
import requests

# Load the base SDXL model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# Define LoRA weights location
LORA_URL = "https://huggingface.co/dennis-brinelinestudios/soulcaller-lora/resolve/main/SDXL_Inkdrawing_Directors_Cut_E.safetensors"
LORA_PATH = "/tmp/SDXL_Inkdrawing_Directors_Cut_E.safetensors"  # Save locally

# Download LoRA weights if not already present
def download_lora():
    response = requests.get(LORA_URL, stream=True)
    if response.status_code == 200:
        with open(LORA_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download LoRA file: {LORA_URL}")

download_lora()

# Load LoRA weights
lora_weights = load_file(LORA_PATH)

# Apply LoRA weights to the model
pipe.unet.load_state_dict(lora_weights, strict=False)

# Function to run predictions
def predict(prompt: str, steps: int = 30):
    return pipe(prompt, num_inference_steps=steps).images[0]

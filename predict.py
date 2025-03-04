from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file
import torch

# Load the base SDXL model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# Define the LoRA repository on Hugging Face
LORA_REPO = "dennis-brinelinestudios/soulcaller-lora"

# Load LoRA weights from Hugging Face
lora_weights = load_file(f"https://huggingface.co/{LORA_REPO}/resolve/main/SDXL_Inkdrawing_Directors_Cut_E.safetensors")

# Apply LoRA weights to the model
pipe.unet.load_state_dict(lora_weights, strict=False)

# Function to run predictions
def predict(prompt: str, steps: int = 30):
    return pipe(prompt, num_inference_steps=steps).images[0]

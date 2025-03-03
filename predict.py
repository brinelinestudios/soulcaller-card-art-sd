import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file

# Load the base SDXL model from Replicate's repository
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# LoRA file (update this with the actual link to your LoRA)
LORA_PATH = "https://huggingface.co/dennis-brinelinestudios/soulcaller-lora/raw/main/SDXL_Inkdrawing_Directors_Cut_E.safetensors"

# Load LoRA weights
lora_weights = load_file(LORA_PATH)

# Apply LoRA to the model
pipe.unet.load_state_dict(lora_weights, strict=False)

# Function to run predictions
def predict(prompt: str, steps: int = 30):
    return pipe(prompt, num_inference_steps=steps).images[0]

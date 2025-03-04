from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file
import torch

# Load the base SDXL model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# Load LoRA weights from Hugging Face
LORA_PATH = "https://huggingface.co/dennis-brinelinestudios/soulcaller-lora/resolve/main/SDXL_Inkdrawing_Directors_Cut_E.safetensors"

# Load LoRA weights properly
lora_weights = load_file(LORA_PATH)

# Apply LoRA weights to the model
pipe.unet.load_state_dict(lora_weights, strict=False)

# Function to run predictions
def predict(prompt: str, steps: int = 30):
    return pipe(prompt, num_inference_steps=steps).images[0]

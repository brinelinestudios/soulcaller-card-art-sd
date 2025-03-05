import os
import requests
from cog import BasePredictor, Input
import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file

class Predictor(BasePredictor):
    def setup(self):
        print("Loading model and LoRA weights...")

        # Set model ID
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = StableDiffusionXLPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        self.pipe.to("cuda")
        print("Model loaded successfully.")

        # Define LoRA weights URL and local path
        LORA_URL = "https://huggingface.co/dennis-brinelinestudios/soulcaller-lora/resolve/main/SDXL_Inkdrawing_Directors_Cut_E.safetensors"
        LORA_PATH = "./SDXL_Inkdrawing_Directors_Cut_E.safetensors"

        # Check if file exists, if not, download it
        if not os.path.exists(LORA_PATH):
            print(f"Downloading LoRA weights from {LORA_URL}...")
            response = requests.get(LORA_URL, stream=True)
            with open(LORA_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("LoRA weights downloaded.")

        # Load the LoRA weights from local file
        print("Loading LoRA weights...")
        lora_weights = load_file(LORA_PATH)
        self.pipe.unet.load_state_dict(lora_weights, strict=False)
        print("LoRA weights loaded.")

    def predict(self, prompt: str = "A test image", steps: int = 30) -> str:
        print(f"Running inference with prompt: {prompt}, steps: {steps}")
        output_image = self.pipe(prompt, num_inference_steps=steps).images[0]

        output_path = "/tmp/output.png"
        output_image.save(output_path)
        print(f"Image saved at {output_path}")

        return output_path

def predict(self, prompt: str = "A test image", steps: int = 30) -> str:
    print(f"🟡 Running inference with prompt: {prompt}, steps: {steps}")

    try:
        output = self.pipe(prompt, num_inference_steps=steps)  # Run model
        print(f"🟢 Model output: {output}")  # Debug: Print full model output

        output_image = output.images[0]  # Extract image
        print("✅ Image generated successfully!")  # Debug: Confirm image exists

        output_path = "/tmp/output.png"
        output_image.save(output_path)  # Save image
        print(f"✅ Image saved at {output_path}")  # Debug: Confirm save success

        return output_path
    except Exception as e:
        print(f"❌ Error during inference: {e}")  # Debug: Print any errors
        return "ERROR"

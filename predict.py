import os
import torch
import requests
import replicate
from cog import BasePredictor, Input
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file

class Predictor(BasePredictor):
    def setup(self):
        """Load the model and LoRA weights into memory for faster predictions"""
        print("Loading model and LoRA weights...")

        # Set model ID
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = StableDiffusionXLPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        self.pipe.to("cuda")

        print("Model loaded successfully.")

        # Define LoRA weights URL and local path
        self.LORA_URL = "https://huggingface.co/dennis-brinelinestudios/soulcaller-lora/resolve/main/SDXL_Inkdrawing_Directors_Cut_E.safetensors"
        self.LORA_PATH = "/tmp/SDXL_Inkdrawing_Directors_Cut_E.safetensors"

        # Check if file exists, if not, download it
        if not os.path.exists(self.LORA_PATH):
            print(f"Downloading LoRA weights from {self.LORA_URL}...")
            response = requests.get(self.LORA_URL, stream=True)
            with open(self.LORA_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("LoRA weights downloaded.")

        # Load the LoRA weights from local file
        print("Loading LoRA weights...")
        lora_weights = load_file(self.LORA_PATH)
        self.pipe.unet.load_state_dict(lora_weights, strict=False)
        print("LoRA weights loaded.")

    def predict(self, prompt: str = Input(default="A test image"), steps: int = Input(default=30)) -> str:
        """Run the model with the given prompt and generate an image"""
        print(f"Running inference with prompt: {prompt}, steps: {steps}")
        output_image = self.pipe(prompt, num_inference_steps=steps).images[0]

        # Save image locally
        output_path = "/tmp/output.png"
        output_image.save(output_path)
        print(f"Image saved at {output_path}")

        # Upload image to Replicate's storage and return the URL
        image_url = replicate.files.upload(output_path)
        print(f"Image available at {image_url}")

        return image_url

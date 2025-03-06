import torch
from diffusers import StableDiffusionXLPipeline
from cog import BasePredictor, Input, Path
from transformers import CLIPImageProcessor
import requests
from io import BytesIO
from PIL import Image
import time

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_URL = "https://huggingface.co/dennis-brinelinestudios/soulcaller-lora/resolve/main/SDXL_Inkdrawing_Directors_Cut_E.safetensors"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory for efficient processing with LoRA support"""
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        )
        self.pipe.to("cuda")
        
        # Load LoRA weights
        self.pipe.load_lora_weights(LORA_URL)

    def predict(
        self,
        prompt: str = Input(description="Input prompt for the model"),
        negative_prompt: str = Input(description="Negative prompt", default=""),
        guidance_scale: float = Input(description="Scale for classifier-free guidance", default=7.5),
        num_inference_steps: int = Input(description="Number of inference steps", default=50),
        seed: int = Input(description="Seed for reproducibility", default=42),
    ) -> Path:
        """Run a prediction"""
        generator = torch.manual_seed(seed)
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        
        output_path = "/tmp/output.png"
        output.save(output_path)
        
        # Upload image to Replicate storage
        try:
            print("🟡 Uploading image to Replicate storage...")
            uploaded_path = Path(output_path).upload()
            time.sleep(1)  # Ensure upload completes

            if uploaded_path:
                print(f"🟢 Uploaded image: {uploaded_path}")
                return uploaded_path  # Return the uploaded image URL
            else:
                print("❌ Upload failed! Returning local path instead.")
                return Path(output_path)  # Return local path as fallback
        except Exception as e:
            print(f"❌ Error uploading image: {e}")
            return Path(output_path)

import torch
from diffusers import StableDiffusionXLPipeline
from cog import BasePredictor, Input, Path
from transformers import CLIPImageProcessor
import requests
from io import BytesIO
from PIL import Image

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_URL = "https://huggingface.co/dennis-brinelinestudios/soulcaller-lora/resolve/main/SDXL_Inkdrawing_Directors_Cut_E.safetensors"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory for efficient processing with LoRA support"""
        print("🟡 Loading Stable Diffusion XL...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        )
        self.pipe.to("cuda")
        print("🟢 Model loaded successfully.")
        
        # Load LoRA weights
        print(f"🟡 Loading LoRA from {LORA_URL}...")
        self.pipe.load_lora_weights(LORA_URL, weight_name="pytorch_lora_weights.safetensors", alpha=1.5)
        
        # Apply LoRA explicitly
        self.pipe.fuse_lora()
        
        # ✅ Debug: Check if LoRA layers are applied
        print(f"🟢 LoRA Layers Loaded: {self.pipe.unet.attn_processors}")

    def predict(
        self,
        prompt: str = Input(description="Input prompt for the model"),
        negative_prompt: str = Input(description="Negative prompt", default=""),
        guidance_scale: float = Input(description="Scale for classifier-free guidance", default=7.5),
        num_inference_steps: int = Input(description="Number of inference steps", default=50),
        seed: int = Input(description="Seed for reproducibility", default=42),
    ) -> Path:
        """Run a prediction"""
        print(f"🟡 Generating image with prompt: {prompt}")
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
        print(f"🟢 Image generated successfully and saved to {output_path}")
        
        # ✅ Upload output to Hugging Face for accessibility
        try:
            with open(output_path, "rb") as img_file:
                response = requests.post("https://huggingface.co/api/upload", files={"file": img_file})
                if response.status_code == 200:
                    print(f"🟢 Uploaded image successfully: {response.json()}")
                else:
                    print(f"❌ Upload failed: {response.text}")
        except Exception as e:
            print(f"❌ Upload error: {e}")
        
        return Path(output_path)
import torch
from diffusers import StableDiffusionXLPipeline
from cog import BasePredictor, Input, Path
from transformers import AutoModel
from huggingface_hub import hf_hub_download
import logging
import os

# Define the base model and LoRA
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_REPO = "dennis-brinelinestudios/soulcaller-lora"  # Correct namespace
LORA_FILENAME = "SDXL_Inkdrawing_Directors_Cut_E.safetensors"

# Enable logging
logging.basicConfig(level=logging.DEBUG)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model and apply LoRA"""
        logging.info("🟡 Loading Stable Diffusion XL Model...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        )
        self.pipe.to("cuda")
        logging.info("🟢 Model loaded successfully.")

        # **Download the LoRA model correctly**
        logging.info(f"🟡 Downloading LoRA weights from {LORA_REPO}...")
        try:
            lora_path = hf_hub_download(repo_id=LORA_REPO, filename=LORA_FILENAME)
            logging.info(f"🟢 LoRA weights downloaded to: {lora_path}")
        except Exception as e:
            logging.error(f"❌ Failed to download LoRA weights: {e}")
            raise e

        # **Load LoRA weights with increased strength**
        logging.info("🟡 Applying LoRA weights with alpha=2.0...")
        self.pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", alpha=2.0)
        logging.info("🟢 LoRA successfully applied.")

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
        logging.info(f"🟡 Running inference: '{prompt}'")
        
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        
        output_path = "/tmp/output.png"
        output.save(output_path)
        logging.info(f"🟢 Image saved to {output_path}")

        return Path(output_path)

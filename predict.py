import torch
from diffusers import StableDiffusionXLPipeline
from cog import BasePredictor, Input, Path
from transformers import AutoModel
from huggingface_hub import hf_hub_download, snapshot_download
import logging
import os

# Define the new fine-tuned model (DreamShaperXL) as default
#MODEL_NAME = "Lykon/DreamShaperXL_Lightning"  # Fine-tuned SDXL for better imagination
MODEL_NAME = "lykon/dreamshaper-xl-1-0"
MODEL_CACHE = "./dreamshaperxl"  # Local cache path for the model

# Keep the old model as a reference (commented out)
# OLD_MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
# OLD_MODEL_CACHE = "./sdxl-model"

# Define the LoRA model
LORA_REPO = "dennis-brinelinestudios/soulcaller-lora"  # Correct namespace
LORA_FILENAME = "SDXL_Inkdrawing_Directors_Cut_E.safetensors"

# Enable logging
logging.basicConfig(level=logging.DEBUG)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model and apply LoRA"""
        logging.info("🟡 Loading DreamShaperXL Model...")
        
        try:
            # Load DreamShaperXL
            if not os.path.exists(MODEL_CACHE):
                logging.info("🟡 Downloading DreamShaperXL...")
                model_path = snapshot_download(repo_id=MODEL_NAME, cache_dir=MODEL_CACHE)
            else:
                logging.info(f"🟢 Using cached model: {MODEL_CACHE}")
                model_path = MODEL_CACHE

            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path, torch_dtype=torch.float16
            )
            self.pipe.to("cuda")
            logging.info("🟢 DreamShaperXL model loaded successfully.")
        
        except Exception as e:
            logging.error(f"❌ Failed to load DreamShaperXL, falling back to Base SDXL: {e}")
            # Fallback to base SDXL model
            model_path = snapshot_download(repo_id="stabilityai/stable-diffusion-xl-base-1.0", cache_dir="./sdxl-model")
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path, torch_dtype=torch.float16
            )
            self.pipe.to("cuda")
            logging.info("🟢 Base SDXL model loaded as fallback.")
        
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
        prompt: str = Input(description="Input prompt for the model", default="A low-detail colored ink drawing of a tcg [Card Type] named: [Carad Name], description: [User Description]"),
        negative_prompt: str = Input(description="Negative prompt", default="excessive shading, detailed background, realism, clutter"),
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

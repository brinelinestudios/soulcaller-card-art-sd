from typing import List
import os
import requests
import time
from cog import BasePredictor, Input, Path
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import load_lora_weights
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

class Predictor(BasePredictor):
    def setup(self):
        """Optimized Model Setup: Faster Boot, Lower Memory Usage"""

        MODEL_CACHE = "./sdxl-model"
        LORA_PATH = "./SDXL_Inkdrawing_Directors_Cut_E.safetensors"
        LORA_URL = "https://huggingface.co/dennis-brinelinestudios/soulcaller-lora/resolve/main/SDXL_Inkdrawing_Directors_Cut_E.safetensors"

        # Load base model from cache or download
        if not os.path.exists(MODEL_CACHE):
            print("🟡 Downloading base model...")
            model_path = snapshot_download(repo_id="stabilityai/stable-diffusion-xl-base-1.0", cache_dir=MODEL_CACHE)
        else:
            print("🟢 Using cached model:", MODEL_CACHE)
            model_path = MODEL_CACHE

        # Load pipeline with optimizations
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            safety_checker=None,
            requires_safety_checker=False,
            variant="fp16",
        )
        self.pipe.vae.enable_tiling()
        self.pipe.to("cuda")
        print("✅ Base model loaded successfully.")

        # Download and load LoRA
        if not os.path.exists(LORA_PATH):
            print(f"🟡 Downloading LoRA weights from {LORA_URL}...")
            response = requests.get(LORA_URL, stream=True)
            with open(LORA_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ LoRA weights downloaded.")
        else:
            print("Using cached lora:")

        try:
            print("🟡 Attempting to load LoRA weights...")
            load_lora_into_pipeline(self.pipe, LORA_PATH)
            print("✅ LoRA weights loaded successfully.")
        except AttributeError as e:
            print(f"❌ LoRA loading failed: {e}")
            print("🔍 Possible Causes: Mismatched LoRA file, incompatible Diffusers version, or missing keys.")
            print("🔧 Try re-downloading or converting the LoRA file.")

    def predict(self, 
                prompt: str = Input(description="Prompt for image generation", default="A test image"),
                steps: int = Input(description="Number of inference steps", default=30)) -> List[Path]:
        """
        Run the image generation model with the given prompt and steps.
        Returns a **publicly accessible URL** of the generated image.
        """
        print(f"🟡 Running inference with prompt: '{prompt}', steps: {steps}")

        try:
            # Generate image
            output = self.pipe(prompt, num_inference_steps=steps)
            print("✅ Model executed successfully.")

            # Extract image from pipeline output
            output_image = output.images[0]
            print("✅ Image extracted from model output.")

            # Define output path
            output_path = "/tmp/output.png"
            output_image.save(output_path)
            print(f"✅ Image saved at {output_path}")

            # ✅ **UPLOAD IMAGE TO REPLICATE STORAGE**
            try:
                print("🟡 Uploading image to Replicate storage...")
                uploaded_path = Path(output_path).upload()
                time.sleep(1)  # Ensure upload completes

                if uploaded_path:
                    print(f"🟢 Uploaded image: {uploaded_path}")
                    return [uploaded_path]  # Return the uploaded image URL
                else:
                    print("❌ Upload failed! Returning local path instead.")
                    return [Path(output_path)]  # Return local path as fallback

            except Exception as e:
                print(f"❌ Exception during upload: {e}")
                return [Path(output_path)]  # Fallback to local path

        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return []

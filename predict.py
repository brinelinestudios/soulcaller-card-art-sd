import os
import requests
from cog import BasePredictor, Input, Path  # Ensure Path is used for returning files
import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file
from huggingface_hub import snapshot_download  # Use snapshot_download to get the full model

class Predictor(BasePredictor):
    def setup(self):
        """Load the base Stable Diffusion XL model and LoRA weights"""
        print("🔵 Setting up the model and LoRA weights...")

        # Download the full SDXL model directory
        print("🟡 Downloading base model from Hugging Face...")
        model_path = snapshot_download(repo_id="stabilityai/stable-diffusion-xl-base-1.0")
        
        # Load base model
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        )
        self.pipe.to("cuda")
        print("🟢 Base model loaded successfully.")

        # Define LoRA weights URL and local path
        LORA_URL = "https://huggingface.co/dennis-brinelinestudios/soulcaller-lora/resolve/main/SDXL_Inkdrawing_Directors_Cut_E.safetensors"
        LORA_PATH = "./SDXL_Inkdrawing_Directors_Cut_E.safetensors"

        # Download LoRA weights if not already present
        if not os.path.exists(LORA_PATH):
            print(f"🟡 Downloading LoRA weights from {LORA_URL}...")
            response = requests.get(LORA_URL, stream=True)
            with open(LORA_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ LoRA weights downloaded.")

        # Load the LoRA weights into the model
        print("🟡 Loading LoRA weights...")
        lora_weights = load_file(LORA_PATH)
        self.pipe.unet.load_state_dict(lora_weights, strict=False)
        print("✅ LoRA weights loaded successfully.")

    def predict(
        self,
        prompt: str = Input(description="Prompt for image generation", default="A test image"),
        steps: int = Input(description="Number of inference steps", default=30)
    ) -> list[Path]:
        """
        Generate an image based on the given prompt and number of steps.
        Returns a list containing the file path of the generated image.
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
            output_image.save(output_path)  # Save image as PNG
            print(f"✅ Image saved at {output_path}")

            # Ensure the image file exists
            if os.path.exists(output_path):
                print("🟢 Image file confirmed to exist.")
            else:
                print("❌ ERROR: Image file not found after saving!")

            # Return the image path as a list of Path objects (Replicate requirement)
            return [Path(output_path)]

        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return []

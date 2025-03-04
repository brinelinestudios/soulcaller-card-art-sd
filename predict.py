from cog import BasePredictor, Path, Input
import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file

class Predictor(BasePredictor):
    def setup(self):
        """Load the model and LoRA weights into memory"""
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = StableDiffusionXLPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        self.pipe.to("cuda")

        # Load LoRA weights
        LORA_PATH = "https://huggingface.co/dennis-brinelinestudios/soulcaller-lora/resolve/main/SDXL_Inkdrawing_Directors_Cut_E.safetensors"
        lora_weights = load_file(LORA_PATH)
        self.pipe.unet.load_state_dict(lora_weights, strict=False)

    def predict(self, prompt: str = Input(description="Prompt for the model"),
                      steps: int = Input(description="Number of inference steps", default=30)) -> Path:
        """Run a single prediction"""
        output_image = self.pipe(prompt, num_inference_steps=steps).images[0]
        
        # Save output image to a file and return the path
        output_path = "/tmp/output.png"
        output_image.save(output_path)
        return Path(output_path)

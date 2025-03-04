from cog import BasePredictor, Path, Input
import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file

class Predictor(BasePredictor):
    def setup(self):
        print("Loading model and LoRA weights...")
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = StableDiffusionXLPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        self.pipe.to("cuda")
        print("Model loaded successfully.")

        # Load LoRA weights
        #LORA_PATH = "https://huggingface.co/dennis-brinelinestudios/soulcaller-lora/resolve/main/SDXL_Inkdrawing_Directors_Cut_E.safetensors"
        LORA_PATH = "./SDXL_Inkdrawing_Directors_Cut_E.safetensors"
        print(f"Loading LoRA weights from {LORA_PATH}...")
        lora_weights = load_file(LORA_PATH)
        self.pipe.unet.load_state_dict(lora_weights, strict=False)
        print("LoRA weights loaded.")

    def predict(self, prompt: str = "A test image", steps: int = 30) -> Path:
        print(f"Running inference with prompt: {prompt}, steps: {steps}")
        output_image = self.pipe(prompt, num_inference_steps=steps).images[0]

        # Save the image
        output_path = "/tmp/output.png"
        output_image.save(output_path)
        print(f"Image saved at {output_path}")

        return Path(output_path)

if __name__ == "__main__":
    print("Initializing Predictor...")
    predictor = Predictor()
    predictor.setup()
    result = predictor.predict()
    print(f"Prediction complete. Image saved at: {result}")

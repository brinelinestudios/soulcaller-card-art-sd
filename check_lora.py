import torch
from safetensors.torch import load_file  # Required for .safetensors format

LORA_PATH = LORA_PATH = "/mnt/c/BrineLineStudios/soulcaller-lora-sd/SDXL_Inkdrawing_Directors_Cut_E.safetensors"
 # Replace with your actual LoRA file path

try:
    state_dict = load_file(LORA_PATH)
    print("🔍 LoRA Keys Found:")
    print(state_dict.keys())  # Should list keys like "lora_te" or "unet"
except Exception as e:
    print(f"❌ ERROR: Could not load LoRA file - {e}")

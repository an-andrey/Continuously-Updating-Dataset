import os
import sys
import torch
from diffusers import AutoPipelineForText2Image
from prompt_manager import CSVPromptStreamer
from dotenv import load_dotenv

load_dotenv()

model_id = sys.argv[1]
target_count = int(sys.argv[2])
model_type = sys.argv[3]
base_model_id = sys.argv[4]

staging_dir = "/home/aandrey/links/scratch/data/staging_images"
os.makedirs(staging_dir, exist_ok=True)

prompt_generator = CSVPromptStreamer()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

try:
    if model_type == "LoRA":
        print(f"Loading Base Model ({base_model_id}) for LoRA injection...")
        pipeline = AutoPipelineForText2Image.from_pretrained(
            base_model_id, 
            torch_dtype=torch_dtype,
            use_safetensors=True,
            requires_safety_checker=False,
            local_files_only=True
        ).to(device)
        
        print(f"Injecting LoRA weights from {model_id}...")
        pipeline.load_lora_weights(model_id, local_files_only=True)
        
    else:
        print(f"Loading standalone model ({model_id})...")
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype,
            use_safetensors=True,
            requires_safety_checker=False,
            local_files_only=True
        ).to(device)
    
    # Calculate loop iterations safely
    iterations = target_count
    if iterations == 0: iterations = 1 # Ensure at least 1 loop runs if testing with small numbers
    
    for i in range(iterations):
        prompt = prompt_generator.get_next_prompt()
        result = pipeline(prompt)
        
        safe_model_name = model_id.replace("/", "_")
        filename = f"hf_{safe_model_name}_{i}.png"
        result.images[0].save(os.path.join(staging_dir, filename))
        
        if i % 100 == 0 and i > 0:
            print(f"Generated {i} images...", flush=True)
            
    print(f"Success! Finished generating for {model_id}", flush=True)
    
except Exception as e:
    print(f"Generation failed: {e}", flush=True)
    sys.exit(1)
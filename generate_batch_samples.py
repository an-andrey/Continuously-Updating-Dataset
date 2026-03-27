import os
import sys
import torch
from diffusers import AutoPipelineForText2Image
from prompt_manager import CSVPromptStreamer
from dotenv import load_dotenv

load_dotenv()

model_id = sys.argv[1]
total_target_count = int(sys.argv[2])
model_type = sys.argv[3]
base_model_id = sys.argv[4]

# Array Logic: Determine this GPU's workload
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

images_per_task = total_target_count // task_count
start_index = task_id * images_per_task
end_index = start_index + images_per_task

# Ensure the final task picks up any remainder due to uneven division
if task_id == task_count - 1:
    end_index = total_target_count

target_count_for_this_gpu = end_index - start_index

staging_dir = "/home/aandrey/links/scratch/data/staging_images"
os.makedirs(staging_dir, exist_ok=True)

prompt_generator = CSVPromptStreamer()

# Fast-forward the prompt generator so it doesn't overlap with other GPUs
for _ in range(start_index):
    prompt_generator.get_next_prompt()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

try:
    if model_type == "LoRA":
        print(f"Task {task_id}: Loading Base Model ({base_model_id}) for LoRA injection...")
        pipeline = AutoPipelineForText2Image.from_pretrained(
            base_model_id, 
            torch_dtype=torch_dtype,
            use_safetensors=True,
            requires_safety_checker=False,
            local_files_only=True
        ).to(device)
        print(f"Task {task_id}: Injecting LoRA weights from {model_id}...")
        pipeline.load_lora_weights(model_id, local_files_only=True)
    else:
        print(f"Task {task_id}: Loading standalone model ({model_id})...")
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype,
            use_safetensors=True,
            requires_safety_checker=False,
            local_files_only=True
        ).to(device)
    
    # Offloads sub-components to standard RAM when not actively being used
    pipeline.enable_model_cpu_offload() 

    # Decodes the batch of images one-by-one at the very end instead of all at once
    pipeline.enable_vae_slicing()
    
    # Batched Generation Loop
    batch_size = 3
    
    for i in range(0, target_count_for_this_gpu, batch_size):
        # Prevent the final batch from generating too many images
        current_batch_size = min(batch_size, target_count_for_this_gpu - i)
        
        prompts = [prompt_generator.get_next_prompt() for _ in range(current_batch_size)]
        results = pipeline(prompts)
        
        for j, img in enumerate(results.images):
            # global_index ensures files are numbered correctly from 0 to 9999 across all nodes
            global_index = start_index + i + j
            safe_model_name = model_id.replace("/", "_")
            filename = f"hf_{safe_model_name}_{global_index}.png"
            img.save(os.path.join(staging_dir, filename))
            
        if i % 100 < batch_size and i > 0:
            print(f"Task {task_id} generated {i}/{target_count_for_this_gpu} images...", flush=True)
            
    print(f"Success! Task {task_id} finished generating for {model_id}", flush=True)
    
except Exception as e:
    print(f"Task {task_id} Generation failed: {e}", flush=True)
    sys.exit(1)
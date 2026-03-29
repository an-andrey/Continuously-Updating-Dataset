import os
import sys
import torch
from diffusers import AutoPipelineForText2Image
from prompt_manager import CSVPromptStreamer
from dotenv import load_dotenv
from datetime import datetime

import traceback

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

EXIT_DATA_FAULT = 10
EXIT_MODEL_FAULT = 11
EXIT_MEMORY_FAULT = 12
EXIT_CODE_BUG = 14

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
    pipeline.enable_vae_tiling()
    
    # Batched Generation Loop
    batch_size = 6
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    for i in range(0, target_count_for_this_gpu, batch_size):
        # Prevent the final batch from generating too many images
        current_batch_size = min(batch_size, target_count_for_this_gpu - i)
        
        prompts = [prompt_generator.get_next_prompt() for _ in range(current_batch_size)]
        results = pipeline(prompts)
        
        for j, img in enumerate(results.images):
            # global_index ensures files are numbered correctly from 0 to 9999 across all nodes
            global_index = start_index + i + j
            safe_model_name = model_id.replace("/", "_")
            filename = f"hf_{safe_model_name}_{today_date}_{global_index}.png"
            img.save(os.path.join(staging_dir, filename))
            
        if i % 100 < batch_size and i > 0:
            print(f"Task {task_id} generated {i}/{target_count_for_this_gpu} images...", flush=True)
            
    print(f"Success! Task {task_id} finished generating for {model_id}", flush=True)
    
except torch.cuda.OutOfMemoryError as e:
    print(f"Task {task_id} VRAM OOM Error: {e}", flush=True)
    sys.exit(EXIT_MEMORY_FAULT)

except Exception as e:
    error_msg = str(e)
    error_msg_lower = error_msg.lower()
    
    # Catch Out of Memory strings (Sometimes thrown outside the specific torch class)
    if "cuda out of memory" in error_msg_lower:
        print(f"Task {task_id} VRAM OOM Error: {e}", flush=True)
        sys.exit(EXIT_MEMORY_FAULT)
        
    # Distinguish File/Path Errors (model.json not found vs typo)
    elif isinstance(e, FileNotFoundError) or isinstance(e, OSError) or "no such file or directory" in error_msg_lower:
        
        # Hugging Face explicitly complains about these files if the repo isn't a valid Diffusers model
        hf_keywords = ["model_index.json", "config.json", "scheduler", "huggingface", "safetensors"]
        
        if any(keyword in error_msg_lower for keyword in hf_keywords):
            print(f"Task {task_id} Model Fault: Missing diffusers config/architecture: {e}", flush=True)
            sys.exit(EXIT_MODEL_FAULT) # Code 11: Blacklist the model
        else:
            print(f"Task {task_id} Data/Path Error (Likely a typo in your paths): {e}", flush=True)
            sys.exit(EXIT_DATA_FAULT) # Code 10: Halt and let you fix the typo

    # Catch generic architecture mismatches
    elif "weight" in error_msg_lower or "pipeline" in error_msg_lower or "expected str" in error_msg_lower:
        print(f"Task {task_id} Model Architecture Error: {e}", flush=True)
        sys.exit(EXIT_MODEL_FAULT) # Code 11: Blacklist the model
        
    # Catch-all for absolute Code Bugs (Syntax, Indentation, TypeErrors)
    else:
        print(f"Task {task_id} CRITICAL CODE BUG:\n{traceback.format_exc()}", flush=True)
        sys.exit(EXIT_CODE_BUG) # Code 14: Kill the coordinator
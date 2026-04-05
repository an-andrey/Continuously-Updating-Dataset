import os
import sys
import torch
from diffusers import AutoPipelineForText2Image
from prompt_manager import CSVPromptStreamer
from dotenv import load_dotenv
from datetime import datetime
import gc

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
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

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
        )
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
        )
    
    #Params to reduce GPU/CPU usage 
    if hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload() 
        print(f"Task {task_id}: CPU offloading enabled. (Safely handling VRAM)", flush=True)
    else:
        pipeline = pipeline.to(device)
        print(f"Task {task_id}: Model manually moved to GPU.", flush=True)

    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
        print(f"Task {task_id}: VAE slicing enabled.", flush=True)

    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
        print(f"Task {task_id}: VAE tiling enabled.", flush=True)
    
    optimal_batch_size = 12 # Start greedy, half it every time cuda out of memory
    images_generated = 0
    pending_prompts = []    # Cache prompts so we don't lose them if a batch fails
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    while images_generated < target_count_for_this_gpu:
        current_chunk_size = min(optimal_batch_size, target_count_for_this_gpu - images_generated)
        
        # Buffer prompts so we don't skip them if the batch fails
        while len(pending_prompts) < current_chunk_size:
            pending_prompts.append(prompt_generator.get_next_prompt())
            
        attempt_prompts = pending_prompts[:current_chunk_size]
        
        try:
            results = pipeline(prompt=attempt_prompts)
            
            for j, img in enumerate(results.images):
                # global_index ensures files are numbered correctly from 0 to 9999 across all nodes
                global_index = start_index + images_generated + j
                safe_model_name = model_id.replace("/", "_")
                filename = f"hf_{safe_model_name}_{today_date}_{global_index}.png"
                img.save(os.path.join(staging_dir, filename))
                
            images_generated += current_chunk_size
            pending_prompts = [] # Clear cached prompts on success
            
            if images_generated % 100 < optimal_batch_size and images_generated > 0:
                print(f"Task {task_id} generated {images_generated}/{target_count_for_this_gpu}... (Locked Batch Size: {optimal_batch_size})", flush=True)

        # Catch PyTorch OOMs *inside* the loop so we can retry instead of crashing
        except torch.cuda.OutOfMemoryError as e:
            print(f"Task {task_id} VRAM Overload at batch {optimal_batch_size}. Flushing memory...", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            
            if optimal_batch_size == 1:
                print(f"Task {task_id} FATAL: Model too large for 1 image.", flush=True)
                sys.exit(EXIT_MEMORY_FAULT)
                
            optimal_batch_size = max(1, optimal_batch_size // 2)
            print(f"Task {task_id}: Halved batch size. Retrying with {optimal_batch_size}...", flush=True)

        except Exception as inner_e:
            # Catch sneaky generic string OOMs, otherwise escalate to the outer try/except block
            if "cuda out of memory" in str(inner_e).lower() or "oom" in str(inner_e).lower():
                print(f"Task {task_id} VRAM Overload at batch {optimal_batch_size}. Flushing memory...", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                
                if optimal_batch_size == 1:
                    print(f"Task {task_id} FATAL: Model too large for 1 image.", flush=True)
                    sys.exit(EXIT_MEMORY_FAULT)
                    
                optimal_batch_size = max(1, optimal_batch_size // 2)
                print(f"Task {task_id}: Halved batch size. Retrying with {optimal_batch_size}...", flush=True)
            else:
                # Escalates things like Architecture Errors to the outer catch
                raise inner_e 

    print(f"Success! Task {task_id} finished generating for {model_id}", flush=True)
    
except Exception as e:
    error_msg = str(e)
    error_msg_lower = error_msg.lower()
    
    if "cuda out of memory" in error_msg_lower:
        print(f"Task {task_id} VRAM OOM Error during init: {e}", flush=True)
        sys.exit(EXIT_MEMORY_FAULT)
        
    elif isinstance(e, FileNotFoundError) or isinstance(e, OSError) or "no such file or directory" in error_msg_lower:
        hf_keywords = ["model_index.json", "config.json", "scheduler", "huggingface", "safetensors"]
        
        if any(keyword in error_msg_lower for keyword in hf_keywords):
            print(f"Task {task_id} Model Fault: Missing diffusers config/architecture: {e}", flush=True)
            sys.exit(EXIT_MODEL_FAULT) 
        else:
            print(f"Task {task_id} Data/Path Error (Likely a typo in your paths): {e}", flush=True)
            sys.exit(EXIT_DATA_FAULT) 

    # FIX 2: Moved architecture mismatch keywords here
    elif any(kw in error_msg_lower for kw in ["weight", "pipeline", "expected str", "time embedding", "incorrect config", "dimension"]):
        print(f"Task {task_id} Model Architecture Error: {e}", flush=True)
        sys.exit(EXIT_MODEL_FAULT) 
        
    else:
        print(f"Task {task_id} CRITICAL CODE BUG:\n{traceback.format_exc()}", flush=True)
        sys.exit(EXIT_CODE_BUG)
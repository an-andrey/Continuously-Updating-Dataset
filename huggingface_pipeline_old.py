import os
import json
import torch
import pandas as pd
import subprocess
import shutil
import time
from dotenv import load_dotenv
import builtins
from datetime import datetime
import sys
import glob

load_dotenv() 

from huggingface_hub import HfApi, model_info, snapshot_download

REGISTRY_FILE = "model_registry.json"
STAGING_DIR = "/home/aandrey/links/scratch/data/staging_images"
TODAY = datetime.now().strftime("%Y-%m-%d")

def timestamped_print(*args, **kwargs):
    """Wraps the standard print function to prepend a timestamp."""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    builtins.print(timestamp, *args, **kwargs)

# Replace the built-in print with custom function
print = timestamped_print

def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_registry(registry):
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=4)

def check_safetensors_available(model_id): #verify safetensors available
    print("verifying tensors...")
    try:
        info = model_info(model_id)
        # Scan all files in the repo for the .safetensors extension
        print(f"scanning through {len(info.siblings)} items")
        file_names = [f.rfilename for f in info.siblings]
        return any(fname.endswith(".safetensors") for fname in file_names)
    except Exception as e:
        print(f"Error checking files for {model_id}: {e}")
        return False


def classify_model(model): 
    '''
    number of generations per category: 
    base: 10k
    fine-tune: 5k
    lora: 2k
    '''
    tags = [tag.lower() for tag in (model.tags or [])]
    card_data = getattr(model, "cardData", {}) or {}
    base_model = card_data.get("base_model", None)
    
    # Handle edge cases where authors improperly format the YAML as a list
    if isinstance(base_model, list) and len(base_model) > 0:
        base_model = base_model[1] # Grab the first element ['base_mode', 'model_id']
        
    # Fallback: Scrape tags if the dependency is injected directly by the Hub
    if not base_model:
        for tag in tags:
            if tag.startswith("base_model:"):
                # Grab ONLY the second half of the split (the actual ID)
                base_model = tag.split(":", 1) 
                break

    safe_base = base_model if base_model else "None"
    
    # Classification based strictly on Hub metadata architecture
    if "lora" in tags or "peft" in tags or "adapter" in tags:
        return "LoRA", 2000, safe_base[1]
        
    # If the model explicitly declares a parent, it is a fine-tune
    if base_model:
        return "Fine-tune", 5000, safe_base[1]
        
    # If it has no parent dependency, it is a root node (Base model)
    return "Base", 10000, "None"

def run_hf_generator():
    api = HfApi()
    
    # Load and sync registry
    registry = load_registry()
    
    print("Fetching txt2img models from Hugging Face...")
    models = api.list_models(filter="diffusers", sort="downloads", full=True, limit=1000)

    for model in models:
        # Skip if already processed successfully or blacklisted
        if model.id in registry:
            status = registry[model.id].get("status")
            if status in ["COMPLETED", "MODEL_FAULT"]:
                continue
            # If INFRASTRUCTURE_FAULT, we let it try again

        downloads = getattr(model, "downloads", 0)
        if downloads < 30000:
            continue
            
        if not check_safetensors_available(model.id):
            registry[model.id] = {"status": "MODEL_FAULT", "reason": "No safetensors found", "date": TODAY}
            save_registry(registry)
            continue

        pipeline_task = getattr(model, "pipeline_tag", "") or ""
        if any(bad_word in pipeline_task.lower() for bad_word in ["video", "audio", "3d"]):
            print(f"Skipping {model.id} because it is a '{pipeline_task}' model.")
            registry[model.id] = {"status": "MODEL_FAULT", "reason": f"Incompatible task: {pipeline_task}", "date": TODAY}
            save_registry(registry)
            continue
            
        model_type, target_count, base_model_id = classify_model(model)
        safe_model_name = model.id.replace("/", "_")

        print(f"\n--- Processing {model.id} ({model_type}) ---")
        
        try:
            print(f"Downloading {model.id}")
            snapshot_download(repo_id=model.id)
            
            if model_type == "LoRA" and base_model_id != "None":
                snapshot_download(repo_id=base_model_id)
            elif model_type == "LoRA" and base_model_id == "None": 
                registry[model.id] = {"status": "MODEL_FAULT", "reason": "Missing base model dependency", "date": TODAY}
                save_registry(registry)
                continue
            
            if target_count >= 10000: array_tasks = 5  
            elif target_count >= 5000: array_tasks = 2  
            else: array_tasks = 1  
                
            print(f"Submitting Array ({array_tasks} tasks) to slurm_logs/{TODAY}/...")
            
            process = subprocess.run([
                "sbatch", 
                "--wait", 
                f"--array=0-{array_tasks-1}", 
                f"--output=data/slurm_logs/{TODAY}/gen_hf-{safe_model_name}-%A_%a.out",
                "submit_generate_batch_samples.sh", 
                model.id, 
                str(target_count), 
                model_type, 
                base_model_id
            ], capture_output=True, text=True)

            # Wait for Lustre to flush the log files to disk
            time.sleep(10)

            if process.returncode == 0:
                print(f"GPU job SUCCESS. Marking {model.id} as COMPLETED.")
                registry[model.id] = {"status": "COMPLETED", "date": TODAY}
            else:
                rc = process.returncode
                print(f"GPU job FAILED with Exit Code: {rc}")

                if rc == 14:
                    print(f"\n[CRITICAL ALERT] Developer Code Bug (Exit 14) detected for {model.id}.")
                    print("HALTING ENTIRE PIPELINE TO PREVENT ENDLESS LOOPING.")
                    sys.exit(1)
                    
                elif rc == 11:
                    registry[model.id] = {"status": "MODEL_FAULT", "reason": "Incompatible Architecture (Exit 11)", "date": TODAY}
                
                elif rc == 12 or rc == 137 or rc == 9:
                    registry[model.id] = {"status": "INFRASTRUCTURE_FAULT", "reason": "Node/GPU Out of Memory", "date": TODAY}
                    
                elif rc == 10:
                    registry[model.id] = {"status": "INFRASTRUCTURE_FAULT", "reason": "Data/Path Error (Exit 10)", "date": TODAY}
                    
                else:
                    # Catch-all for generic Slurm submission rejections (Usually Code 1 or 255)
                    registry[model.id] = {"status": "INFRASTRUCTURE_FAULT", "reason": f"Generic Slurm Rejection (Code {rc})", "date": TODAY}
            
            save_registry(registry)
            
            # Cleanup
            print(f"Cleaning up {model.id} from Hugging Face cache...")
            safe_dir_name = "models--" + model.id.replace("/", "--")
            model_path = os.path.join(os.environ.get("HF_HOME"), "hub", safe_dir_name)
            if os.path.exists(model_path):
                shutil.rmtree(model_path)

        except KeyboardInterrupt: 
            print("Keyboard interruption, halting.")
            sys.exit(0)

        except Exception as e:
            print(f"Unexpected Pipeline Failure for {model.id}: {e}")
            registry[model.id] = {"status": "INFRA_FAULT", "reason": str(e), "date": TODAY}
            save_registry(registry)

if __name__ == "__main__":
    run_hf_generator()



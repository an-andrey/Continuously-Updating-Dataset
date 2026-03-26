import os
import json
import torch
import pandas as pd
import subprocess
import shutil
from dotenv import load_dotenv

load_dotenv() 

from huggingface_hub import HfApi, model_info, snapshot_download

TRACKER_FILE = "seen_models.json"
FAILED_FILE = "failed_models.json"

def load_json_registry(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return []

def save_json_registry(filepath, data_list):
    with open(filepath, 'w') as f:
        json.dump(data_list, f)

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
        base_model = base_model
        
    # Fallback: Scrape tags if the dependency is injected directly by the Hub
    if not base_model:
        for tag in tags:
            if tag.startswith("base_model:"):
                base_model = tag.split(":", 1)
                break

    safe_base = base_model if base_model else "None"
    
    # Classification based strictly on Hub metadata architecture
    if "lora" in tags or "peft" in tags or "adapter" in tags:
        return "LoRA", 2000, safe_base[1] # safe_base stores as array ['base_model', 'model_id']
        
    # If the model explicitly declares a parent, it is a derivative
    if base_model:
        return "Fine-tune", 5000, safe_base
        
    # If it has no parent dependency, it is a root node (Base model)
    return "Base", 10000, "None"

def run_hf_generator():
    api = HfApi()
    seen_models = load_json_registry(TRACKER_FILE)
    failed_models = load_json_registry(FAILED_FILE)
    
    print("Fetching txt2img models from Hugging Face...")
    models = api.list_models(
        filter="diffusers", 
        sort="downloads", 
        full=True, 
        limit=1000
        )

    for model in models:
        downloads = getattr(model, "downloads", 0)
        if downloads < 30000 or model.id in seen_models or model.id in failed_models or not check_safetensors_available(model.id):
            continue

        pipeline_task = getattr(model, "pipeline_tag", "") or ""
        
        # If the word "video", "audio", or "3d" is anywhere in the task, skip it
        if any(bad_word in pipeline_task.lower() for bad_word in ["video", "audio", "3d"]):
            print(f"Skipping {model.id} because it is a '{pipeline_task}' model.")
            failed_models.append(model.id)
            save_json_registry(FAILED_FILE, failed_models)
            continue
            
        model_type, target_count, base_model_id = classify_model(model)

        print(f"\n--- Processing {model.id}, that's a {model_type} model ---")

        print(model_type, target_count, base_model_id)
        
        try:
            print(f"Downloading {model.id}")
            snapshot_download(repo_id=model.id)
            
            if model_type == "LoRA" and base_model_id != "None":
                print(f"Downloading required Base Model ({base_model_id})...")
                snapshot_download(repo_id=base_model_id)
            
            elif model_type == "LoRA" and base_model_id == "None": 
                print("lora model had no base model associated, skipping it")
                failed_models.append(model.id)
                save_json_registry(FAILED_FILE, failed_models)
                continue
            
            if target_count >= 10000:
                array_tasks = 5  # Splits 10k into 5 nodes x 2000
            elif target_count >= 5000:
                array_tasks = 2  # Splits 5k into 2 nodes x 2500
            else:
                array_tasks = 1  # Runs 2k on 1 node
                
            print(f"Submitting GPU job array ({array_tasks} parallel tasks) and waiting...")
            
            # The --wait flag will pause the coordinator until ALL array tasks complete
            process = subprocess.run([
                "sbatch", 
                "--wait", 
                f"--array=0-{array_tasks-1}", 
                "submit_generate_batch_samples.sh", 
                model.id, 
                str(target_count), 
                model_type, 
                base_model_id
            ])

            if process.returncode == 0:
                print(f"GPU job successful. Logging {model.id}.")
                seen_models.append(model.id)
                save_json_registry(TRACKER_FILE, seen_models)
            else:
                print(f"ERROR: GPU job for {model.id} failed. It will not be marked as seen.")
                failed_models.append(model.id)
                save_json_registry(FAILED_FILE, failed_models)
            
            print(f"GPU job finished. Cleaning up {model.id} to save space...")
            safe_dir_name = "models--" + model.id.replace("/", "--")
            model_path = os.path.join(os.environ.get("HF_HOME"), "hub", safe_dir_name)

            if os.path.exists(model_path):
                print(f"model found at {model_path} and will be removed")
                shutil.rmtree(model_path)
                
            seen_models.append(model.id)
            save_json_registry(TRACKER_FILE, seen_models)
        
        except KeyboardInterrupt: 
            print("Keyboard interruption, deleting the model")
            safe_dir_name = "models--" + model.id.replace("/", "--")
            model_path = os.path.join(os.environ.get("HF_HOME"), "hub", safe_dir_name)

            if os.path.exists(model_path):
                print(f"model found at {model_path} and will be removed")
                shutil.rmtree(model_path)

        except Exception as e:
            print(f"Failed pipeline for {model.id}: {e}")

if __name__ == "__main__":
    run_hf_generator()



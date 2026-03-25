import os
import json
import torch
import pandas as pd
from diffusers import AutoPipelineForText2Image
from huggingface_hub import HfApi, model_info, snapshot_download
from prompt_manager import CSVPromptStreamer
import subprocess
import shutil
from dotenv import load_dotenv

load_dotenv()
TRACKER_FILE = "seen_models.json"

def load_seen_models(): # json file holds every model that's been prompted already
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r') as f:
            return json.load(f)
    return []

def save_seen_models(seen_list): # write to json file
    with open(TRACKER_FILE, 'w') as f:
        json.dump(seen_list, f)

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
        return "LoRA", 2000, safe_base
        
    # If the model explicitly declares a parent, it is a derivative
    if base_model:
        return "Fine-tune", 5000, safe_base
        
    # If it has no parent dependency, it is a root node (Base model)
    return "Base", 10000, "None"

def run_hf_generator():
    api = HfApi()
    seen_models = load_seen_models()
    
    print("Fetching txt2img models from Hugging Face...")
    models = api.list_models(
        filter="diffusers", 
        sort="downloads", 
        full=True, 
        limit=5
        )

    for model in models:
        downloads = getattr(model, "downloads", 0)
        if downloads < 30000 or model.id in seen_models or not check_safetensors_available(model.id):
            continue
            
        model_type, target_count, base_model_id = classify_model(model)
        print(f"\n--- Processing {model.id}, that's a {model_type} model ---")
        
        try:
            print(f"Downloading {model.id}")
            snapshot_download(repo_id=model.id, allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt"])
            
            if model_type == "LoRA" and base_model_id != "None":
                print(f"Downloading required Base Model ({base_model_id})...")
                snapshot_download(repo_id=base_model_id, allow_patterns=["**/*.safetensors", "**/*.json", "**/*.txt"])
            
            elif model_type == "LoRA" and base_model_id == "None": 
                print("lora model had no base model associated, skipping it")
                continue

            print(f"Submitting GPU job and waiting...")
            t = int(target_count/1000)
            process = subprocess.run(["sbatch", "--wait", "submit_generate_samples.sh", model.id, str(t), model_type, base_model_id])

            if process.returncode == 0:
                print(f"GPU job successful. Logging {model.id}.")
                seen_models.append(model.id)
                save_seen_models(seen_models)
            else:
                print(f"ERROR: GPU job for {model.id} failed. It will not be marked as seen.")
            
            print(f"GPU job finished. Cleaning up {model.id} to save space...")
            safe_dir_name = "models--" + model.id.replace("/", "--")
            model_path = os.path.join("models/", "hub", safe_dir_name)
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                
            seen_models.append(model.id)
            save_seen_models(seen_models)
            
        except Exception as e:
            print(f"Failed pipeline for {model.id}: {e}")

if __name__ == "__main__":
    run_hf_generator()



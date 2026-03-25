import os
import json
import torch
import pandas as pd
from diffusers import AutoPipelineForText2Image
from huggingface_hub import HfApi, model_info
from prompt_manager import CSVPromptStreamer

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
    try:
        info = model_info(model_id)
        # Scan all files in the repo for the .safetensors extension
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
    
    # 1. Check for LoRA/Adapter first (LoRAs can also declare a base_model)
    if "lora" in tags or "peft" in tags or "adapter" in tags:
        return "LoRA", 2000
        
    # 2. Check the official Model Tree metadata
    # We safely get the cardData dictionary, defaulting to empty if missing
    card_data = getattr(model, "cardData", {}) or {}
    base_model = card_data.get("base_model", None)
    
    if base_model: #if base model is declared, then it's a fine-tune
        return "Fine-tune", 5000

    # allback for older models missing proper YAML metadata
    author = model.id.split('/')[0].lower()
    base_authors = ["stabilityai", "runwayml", "black-forest-labs", "compvis", "qwen", "google", "meta"]
    
    if "base_model" in tags or author in base_authors:
        return "Base", 10000
        
    # If it has no base_model metadata and isn't a known base author
    return "Fine-tune", 5000


def run_hf_generator(staging_dir="data/staging"): # main loop, called in main.py
    api = HfApi()
    print("shuffling csv with prompts")
    prompt_generator = CSVPromptStreamer()
    print("loading seen models")
    seen_models = load_seen_models()
    
    print("Fetching txt2img models from Hugging Face (at least 30k downloads)")

    models = api.list_models(
        filter="diffusers", # filtering by diffusion instead of txt2img to catch multimodal models
        sort="downloads",
        direction=-1,
        expand=["downloadsAllTime","downloads"],
        cardData=True,
        limit=200
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        print("GPU detected")
    else: 
        print("running on CPU")
        
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model_list = []

    for model in models:
        downloads = model.downloads_all_time or 0

        if downloads < 30000 or model.id in seen_models:
            continue

        if not check_safetensors_available(model.id): # removing models w/out safetensors
            continue
            
        model_type, target_count = classify_model(model)
        print(f"Processing {model.id} | Type: {model_type} | Target: {target_count} images")
        model_list.append([model.id, model_type, target_count, downloads])
        
        try:
            pipeline = AutoPipelineForText2Image.from_pretrained(
                model.id, 
                torch_dtype=torch_dtype,
                use_safetensors=True,
                requires_safety_checker=False 
            ).to(device)
            
            # Generate the target amount of images
            for i in range(int(target_count)/1000): # doing much less for testing for now
                prompt = prompt_generator.get_next_prompt()
                result = pipeline(prompt)
                
                safe_model_name = model.id.replace("/", "_")
                filename = f"hf_{safe_model_name}_{i}.png"
                filepath = os.path.join(staging_dir, filename)
                
                result.images[0].save(filepath)
                
                if i % 100 == 0 and i > 0:
                    print(f"generated {i}/{target_count}")
                    
            # Free memory
            del pipeline
            if device == "cuda":
                torch.cuda.empty_cache()
                
            # Mark as seen only if successful
            seen_models.append(model.id)
            
        except Exception as e:
            print(f"Failed to process {model.id}: {e}")
    
    return pd.DataFrame(model_list, columns=("Model", "Type", "Target", "Downloads"))

model_list = run_hf_generator()
model_list.to_csv("models.csv", index=False)
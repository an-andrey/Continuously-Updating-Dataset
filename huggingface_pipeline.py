import os
import json
import torch
import pandas as pd
from diffusers import AutoPipelineForText2Image
from huggingface_hub import HfApi, model_info
from prompt_manager import ReLaionPromptStreamer, MidjourneyPromptStreamer

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


def classify_model(model): #Classifies model (Lora, base or fine-tune) and returns the required generation count
    tags = [tag.lower() for tag in (model.tags or [])]
    author = (model.id.split('/')[0]).lower()
    
    # Known base model authors (ask victor for more)
    base_authors = ["stabilityai", "runwayml", "black-forest-labs", "compvis", "qwen"]
    
    if "lora" in tags:
        return "LoRA", 2000
    elif "base_model" in tags or author in base_authors:
        return "Base", 10000
    else: #no way to detect fine-tune, making it as a fall-back
        return "Fine-tune", 5000

import pandas as pd

def run_hf_generator(staging_dir="data/staging"): # main loop, called in main.py
    api = HfApi()
    prompt_generator = ReLaionPromptStreamer()
    seen_models = load_seen_models()
    
    print("Fetching txt2img models from Hugging Face (at least 30k downloads)")

    models = api.list_models(
        library="diffusers", # filtering by duffusion instead of txt2img to catch multimodal models
        sort="downloads",
        direction=-1,
        expand=["downloadsAllTime","downloads"],
        limit=10
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        
        try:
            pipeline = AutoPipelineForText2Image.from_pretrained(
                model.id, 
                torch_dtype=torch_dtype,
                use_safetensors=True,
                requires_safety_checker=False 
            ).to(device)
            
            # Generate the target amount of images
            # for i in range(target_count):
            #     prompt = prompt_generator.get_next_prompt()
            #     result = pipeline(prompt)
                
            #     safe_model_name = model.id.replace("/", "_")
            #     filename = f"hf_{safe_model_name}_{i}.png"
            #     filepath = os.path.join(staging_dir, filename)
                
            #     result.images[0].save(filepath)
                
            #     if i % 100 == 0 and i > 0:
            #         print(f"generated {i}/{target_count}")
                    
            # Free memory
            del pipeline
            if device == "cuda":
                torch.cuda.empty_cache()
                
            # Mark as seen only if successful
            seen_models.append(model.id)
            model_list.append([model.id, model_type, target_count, downloads])
            
        except Exception as e:
            print(f"Failed to process {model.id}: {e}")
    
    return pd.DataFrame(model_list, columns=("Model", "Type", "Target", "Downloads"))

model_list = run_hf_generator()
model_list.to_csv("models.csv", index="False")
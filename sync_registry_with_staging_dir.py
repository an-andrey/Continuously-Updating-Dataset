import os
import json
import re
from datetime import datetime

STAGING_DIR = "/home/aandrey/links/scratch/data/staging_images"
REGISTRY_FILE = "model_registry.json"
TODAY = datetime.now().strftime("%Y-%m-%d")

def build_initial_registry():
    print(f"Scanning {STAGING_DIR} for existing images. This may take a minute...")
    registry = {}
    
    if not os.path.exists(STAGING_DIR):
        print("Staging directory not found. Exiting.")
        return

    # Regex to strip the index (e.g., "_123") OR the date+index (e.g., "_2026-03-28_123") from the end
    tail_pattern = re.compile(r'(_\d{4}-\d{2}-\d{2})?_\d+$')

    found_count = 0
    for entry in os.scandir(STAGING_DIR):
        if entry.is_file() and entry.name.startswith("hf_") and entry.name.endswith(".png"):
            # 1. Strip 'hf_' from start and '.png' from end
            core_name = entry.name[3:-4] 
            
            # 2. Strip the trailing numbers and/or dates
            safe_name = tail_pattern.sub('', core_name)
            
            # 3. Convert safe name back to Hugging Face ID (replace ONLY the first underscore with a slash)
            original_id = safe_name.replace("_", "/", 1)
            
            if original_id not in registry:
                registry[original_id] = {
                    "status": "COMPLETED",
                    "date": TODAY,
                    "note": "Imported from historical scratch data"
                }
                found_count += 1
                
            # Optional: Print progress so you know it hasn't frozen
            if found_count % 50 == 0:
                print(f"Discovered {found_count} unique models...", end='\r')

    print(f"\nScan complete. Found {len(registry)} unique successful models.")
    
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=4)
        
    print(f"Successfully generated clean {REGISTRY_FILE}.")

if __name__ == "__main__":
    build_initial_registry()
from huggingface_hub import HfApi
import pandas as pd

api = HfApi()

min_downloads = 30_000

model_amt_limit = 10

models = api.list_models(
    filter="text-to-image",
    sort="created_at",      
    expand=["safetensors"], # Required to access parameter metadata
    limit = 100
)

model_data = []

for model in models:
    downloads = model.downloads_all_time or 0
    
    if downloads >= min_downloads:
        total_params = 0
        if hasattr(model, "safetensors") and model.safetensors:
            total_params = model.safetensors.get("parameters", {}).get("total", 0)

        model_data.append({
            "id": model.id,
            "author": model.author,
            "parameters": total_params, 
            "downloads": downloads,
            "likes": model.likes,
            "created_at": model.created_at 
        })
        
        if len(model_data) == model_amt_limit:
            break

df = pd.DataFrame(model_data)

if df.shape != (0,0):
    df['created_at'] = pd.to_datetime(df['created_at'])

    print(df.head())

else: 
    print("no models found")
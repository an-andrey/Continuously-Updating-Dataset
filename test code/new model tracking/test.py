from huggingface_hub import HfApi
import pandas as pd

api = HfApi()

# Use 'trendingScore' instead of 'createdAt'
# This automatically skips the 85k 'empty' models and finds recent ones people actually use
models = api.list_models(
    filter="text-to-image",
    sort="created_at",
    expand=["safetensors", "downloads"]
)

model_data = []

for model in models:

    total_params = 0
    if hasattr(model, "safetensors") and model.safetensors:
        total_params = model.safetensors.get("parameters", {}).get("total", 0)

    model_data.append({
        "id": model.id,
        "downloads": model.downloads,
        "parameters": total_params,
        "created_at": getattr(model, "created_at", None),
    })

# Create and sort by date locally
df = pd.DataFrame(model_data)
if not df.empty:
    df['created_at'] = pd.to_datetime(df['downloads'])
    # Now sort by date locally so the newest of the 'trending' ones are on top
    df = df.sort_values(by="created_at", ascending=False)
    print(df.head(10))
else:
    print("Still no models found. Try lowering the download threshold or removing the task filter.")
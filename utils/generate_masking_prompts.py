import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

LLM_MODEL_HF_ID = "google/gemma-2b-it"
DATASET_ID = "voxel51/open-images-v7"
OUTPUT_CSV = "data/prompts/masked_prompts.csv"

def load_llm_model(model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    
    if device == "cpu":
        model = model.to(device)

    return model, tokenizer, device

def generate_inpainting_prompt(model, tokenizer, device, image_description, mask_label):
    system_instruction = (
        "You are a helpful assistant. Provide a single, short caption to replace the specified "
        "object in an image. Output only the prompt, with no surrounding text."
    )
    user_message = (
        f"An image is described as: '{image_description}'. "
        f"I want to replace the object labeled as '{mask_label}'. "
        f"Provide a short, creative text prompt describing a fitting replacement for this object."
    )

    messages = [
        {"role": "user", "content": f"{system_instruction}\n\n{user_message}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )
    
    response = tokenizer.decode(outputs[inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return response.strip()

def build_prompt_dataset(output_csv, sample_limit=1000):
    model, tokenizer, device = load_llm_model(LLM_MODEL_HF_ID)

    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    
    output_data = []
    processed_count = 0

    for item in dataset:
        if processed_count >= sample_limit:
            break

        # Extract metadata. Adjust keys based on the exact dataset structure.
        image_id = item.get("id", "unknown_id")
        image_url = item.get("image_url", "")
        
        # Open Images contains multiple labels/masks. Select the first valid one.
        segmentations = item.get("segmentations", [])
        if not segmentations:
            continue
            
        selected_mask = segmentations
        mask_label = selected_mask.get("label", "object")
        mask_url = selected_mask.get("mask_url", "")
        
        # Open Images often lacks full captions. You may need to concatenate labels 
        # or use a pre-captioned subset.
        image_description = item.get("caption", f"An image containing {mask_label}")

        generated_prompt = generate_inpainting_prompt(
            model, 
            tokenizer, 
            device, 
            image_description, 
            mask_label
        )

        output_data.append({
            "image_id": image_id,
            "image_url": image_url,
            "mask_url": mask_url,
            "original_description": image_description,
            "mask_label": mask_label,
            "generated_prompt": generated_prompt
        })

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count} samples.")

    df = pd.DataFrame(output_data)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {processed_count} prompts to {output_csv}")

if __name__ == "__main__":
    build_prompt_dataset()
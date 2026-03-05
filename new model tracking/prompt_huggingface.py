import os
import torch
from diffusers import AutoPipelineForText2Image

OUTPUT_DIRECTORY = "data/open_source"

def generate_and_save_image(model_id, prompt):
    """
    Downloads/loads a Hugging Face text-to-image model, generates an image based on a prompt,
    and saves it to the specified output directory.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # Determine hardware capability (Use GPU if available for faster generation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set data type to float16 for GPU to save VRAM, otherwise use float32 for CPU
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Load the model pipeline
    print(f"Loading model weights for '{model_id}'...")
    print("(If this is your first time using this model, it will download and cache the weights locally.)")
    
    try:
        # AutoPipeline automatically detects the correct architecture
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype,
            use_safetensors=True 
        )
        pipeline = pipeline.to(device)
    except Exception as e:
        print(f"\nFailed to load model '{model_id}'.")
        print(f"Error: {e}")
        return
        
    # 4. Generate the image
    print(f"\nGenerating image for prompt: '{prompt}'")
    try:
        # The pipeline returns an object containing a list of generated PIL images
        result = pipeline(prompt)
        image = result.images[0]
    except Exception as e:
        print(f"\nFailed to generate image.")
        print(f"Error: {e}")
        return
        
    # Save the output in the output directory
    # Replace slashes in the model ID so it doesn't create unwanted subdirectories
    safe_model_name = model_id.replace("/", "_")
    filename = f"{safe_model_name}_output.png"
    filepath = os.path.join(OUTPUT_DIRECTORY, filename)
    
    image.save(filepath)
    print(f"\nSuccess! Image saved locally to: {filepath}")


#TESTING - will call from huggingface_tracker otherwise
if __name__ == "__main__":
    target_model_id = "runwayml/stable-diffusion-v1-5" 
    
    user_prompt = "A frog holding maracas on a lilypad"
    
    generate_and_save_image(
        model_id=target_model_id, 
        prompt=user_prompt
    )
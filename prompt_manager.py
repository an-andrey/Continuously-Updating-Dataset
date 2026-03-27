# import random
# from datasets import load_dataset

# class ReLaionPromptStreamer:
#     def __init__(self, dataset_path="supermodelresearch/Re-LAION-Caption19M"):
#         print(f"Initializing stream from {dataset_path}")
#         # We use streaming=True to avoid downloading entire dataset
#         self.dataset = load_dataset(dataset_path, split="train", streaming=True)
#         self.shuffled_stream = self.dataset.shuffle(seed=random.randint(0, 100_000), buffer_size=10_000)
#         self.iterator = iter(self.shuffled_stream)

#     def get_next_prompt(self):
#         try:
#             sample = next(self.iterator)
#             return sample['caption']
        
#         except StopIteration:
#             # If we hit the end, restart the stream
#             self.iterator = iter(self.shuffled_stream)
#             return self.get_next_prompt()

import pandas as pd
import random

class CSVPromptStreamer:
    def __init__(self, filepath="unused_prompts2.csv"):
        print(f"Loading and shuffling prompts from {filepath}...")
        
        # Load the CSV
        df = pd.read_csv(filepath)
        
        # Grab the column that contains the prompt
        self.prompts = df["prompt"].dropna().tolist()
        
        # Shuffle immediately
        random.shuffle(self.prompts)
        
        self.index = 0
        self.total_prompts = len(self.prompts)
        print(f"Successfully loaded {self.total_prompts} prompts into memory.")

    def get_next_prompt(self):
        # Reshuffle if we hit the end of the list
        if self.index >= self.total_prompts:
            random.shuffle(self.prompts)
            self.index = 0
            
        prompt = self.prompts[self.index]
        self.index += 1
        
        return str(prompt)
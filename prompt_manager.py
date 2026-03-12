import random
from datasets import load_dataset

class ReLaionPromptStreamer:
    def __init__(self, dataset_path="supermodelresearch/Re-LAION-Caption19M"):
        print(f"Initializing stream from {dataset_path}")
        # We use streaming=True to avoid downloading entire dataset
        self.dataset = load_dataset(dataset_path, split="train", streaming=True)
        self.shuffled_stream = self.dataset.shuffle(seed=random.randint(0, 100_000), buffer_size=10_000)
        self.iterator = iter(self.shuffled_stream)

    def get_next_prompt(self):
        try:
            sample = next(self.iterator)
            return sample['caption']
        
        except StopIteration:
            # If we hit the end, restart the stream
            self.iterator = iter(self.shuffled_stream)
            return self.get_next_prompt()

# prompt_engine = ReLaionPromptStreamer()

# for i in range(15):
#     print(prompt_engine.get_next_prompt())

import random

class MidjourneyPromptStreamer: #https://www.kaggle.com/datasets/nikbearbrown/one-million-random-midjourney-prompts?resource=download&select=openai_Udxg6_etsy_prompts.csv
    def __init__(self, filepath = "data/openai_Udxg6_etsy_prompts.csv"):
        self.filepath = filepath
        self.generator = self.create_generator()

    def create_generator(self): 
        with open(self.filepath, 'r') as f: 
            while True: 
                row = f.readline()
                yield row.split(',')[0]
                
                i += 1

                if i >= max: 
                    self.prompt_list = self.shuffle_prompts()
                    i = 0

    def get_next_prompt(self):
        return next(self.generator)

# prompt_gen = MidjourneyPromptStreamer()
# for i in range(15):
#     print(prompt_gen.get_next_prompt())
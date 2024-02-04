import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "tokyotech-llm/Swallow-7b-instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

with open(input_file_path, 'r') as f:
    data = f.read()

print(f"length of dataset in characters: {len(data):,}")

tokens = tokenizer(data)
# print(tokens)
# print(tokenizer.convert_ids_to_tokens(tokens['input_ids']))

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

print(train_data[:10])

train_ids = tokenizer(train_data)['input_ids']
val_ids = tokenizer(val_data)['input_ids']
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

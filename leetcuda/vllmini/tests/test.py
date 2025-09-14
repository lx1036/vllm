

from transformers import GPT2Tokenizer
import torch

model_path = "/work/cache/gpt2"
prompt = "who are you"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokens = tokenizer.encode(prompt) # [7454, 2402, 257, 640]
input_ids = torch.tensor([tokens], dtype=torch.int64, device=device)
print(input_ids.size(), input_ids) # torch.Size([1, 3]) tensor([[8727,  389,  345]], device='cuda:0')
tokens = tokenizer.decode(input_ids[0])
print(tokens) # who are you


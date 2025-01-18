import torch
from sae_library import register_pali_gemma
from transformers import AutoTokenizer

from transformer_lens import HookedTransformer

register_pali_gemma()

device = "cuda"
model_id = "google/paligemma2-3b-pt-896"

# Now we can use AutoModelForCausalLM
model = HookedTransformer.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
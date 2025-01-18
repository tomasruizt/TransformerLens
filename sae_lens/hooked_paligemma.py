import torch
from transformer_lens import HookedTransformer
from sae_library import test_heavens_and_earth_prompt
from sae_library import register_pali_gemma

torch.set_grad_enabled(False)  # avoid blowing up mem
device = "cuda"

register_pali_gemma()

model_id = "google/paligemma2-3b-pt-896"
model = HookedTransformer.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model.to(device)

print(model.to_str_tokens("hello world", prepend_bos=True))
print(model.to_str_tokens("hello world", prepend_bos=False))
print(model.to_tokens("this is a longer sentence"))
test_heavens_and_earth_prompt(model=model)

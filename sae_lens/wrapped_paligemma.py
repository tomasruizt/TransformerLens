import torch
from sae_library import test_heavens_and_earth_prompt
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from sae_library import GemmaWrapper

torch.set_grad_enabled(False)  # avoid blowing up mem
device = "cuda"

model_id = "google/paligemma2-3b-pt-896"
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
).eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

pgmodel2 = GemmaWrapper(model, processor.tokenizer, device)
print(pgmodel2.to_str_tokens("hello world", prepend_bos=True))
print(pgmodel2.to_str_tokens("hello world", prepend_bos=False))
print(pgmodel2.to_tokens("this is a longer sentence", prepend_bos=True))
test_heavens_and_earth_prompt(model=pgmodel2)

from dataclasses import dataclass
from typing import TypeAlias
import torch
from transformers import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    Gemma2ForCausalLM,
    GemmaTokenizerFast,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
from transformer_lens.utils import test_prompt

PG: TypeAlias = PaliGemmaForConditionalGeneration


def register_pali_gemma() -> None:
    @property
    def get_model(self: PG):
        return self.language_model.model

    @property
    def get_lm_head(self: PG):
        return self.language_model.lm_head

    PG.model = get_model
    PG.lm_head = get_lm_head

    # Register PaliGemma with the Auto classes
    CONFIG_MAPPING.register("pali_gemma2", PaliGemmaConfig)
    MODEL_FOR_CAUSAL_LM_MAPPING.register(PaliGemmaConfig, PG)


def test_heavens_and_earth_prompt(model: PG) -> None:
    prompt = "In the beginning, God created the heavens and the"
    answer = "earth"

    # Show that the model can confidently predict the next token.
    test_prompt(prompt, answer, model, prepend_bos=True)


@dataclass
class GemmaWrapper:
    model: Gemma2ForCausalLM
    tokenizer: GemmaTokenizerFast
    device: "str"

    def __getattr__(self, name):
        return getattr(self.model, name)

    def to_tokens(self, text: str, prepend_bos: bool) -> torch.Tensor:
        if isinstance(text, list):
            text = text[0]
        prev = self.tokenizer.add_bos_token
        self.tokenizer.add_bos_token = prepend_bos
        tokens = self.tokenizer.encode(
            text, return_tensors="pt", add_special_tokens=prepend_bos
        )
        self.tokenizer.add_bos_token = prev
        return tokens.to(self.device)

    def to_str_tokens(self, text: str, prepend_bos: bool) -> list[str]:
        ids: torch.Tensor = self.to_tokens(text, prepend_bos)
        assert ids.shape[0] == 1, "Batch size must be 1"
        ids = ids[0]
        return [self.tokenizer.decode(id) for id in ids.tolist()]

    def to_string(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens)

    def __call__(self, tokens):
        outs = self.model(tokens)
        return outs["logits"]

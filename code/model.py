from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelHandler:
    def __init__(self, model_config):
        self.base_model = model_config["base_model"]
        self.torch_dtype = getattr(torch, model_config["torch_dtype"])
        self.tokenizer_config = model_config["tokenizer"]

    def setup(self):
        model = self._load_model()
        tokenizer = self._load_tokenizer()
        return model, tokenizer

    def _load_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.base_model, torch_dtype=self.torch_dtype, trust_remote_code=True
        )

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self._setup_tokenizer(tokenizer)
        logger.debug(tokenizer.chat_template)
        return tokenizer

    def _setup_tokenizer(self, tokenizer):
        tokenizer.chat_template = self.tokenizer_config["chat_template"]
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = self.tokenizer_config["padding_side"]

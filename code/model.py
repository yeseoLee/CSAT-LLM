import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelHandler:
    def __init__(self, model_config):
        self.base_model = model_config["base_model"]
        self.model_config = model_config["model"]
        self.tokenizer_config = model_config["tokenizer"]

    def setup(self):
        model = self._load_model()
        tokenizer = self._load_tokenizer()
        return model, tokenizer

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            torch_dtype=getattr(torch, self.model_config["torch_dtype"]),
            low_cpu_mem_usage=self.model_config["low_cpu_mem_usage"],
            load_in_8bit=self.model_config["load_in_8bit"],
            load_in_4bit=self.model_config["load_in_4bit"],
        )
        return model

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self._setup_tokenizer(tokenizer)
        return tokenizer

    def _setup_tokenizer(self, tokenizer):
        tokenizer.chat_template = self.tokenizer_config["chat_template"]
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = self.tokenizer_config["padding_side"]

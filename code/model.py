from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
        torch_dtype = getattr(torch, self.model_config["torch_dtype"])
        base_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": self.model_config["low_cpu_mem_usage"]}

        if self.model_config["quantization"] == "BitsAndBytes":
            bits = self.model_config["bits"]
            if bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=self.model_config["use_double_quant"],
                    bnb_8bit_compute_dtype=torch_dtype,
                )
            elif bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=self.model_config["use_double_quant"],
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            else:
                raise ValueError(f"Unsupported bits value: {bits}")

            base_kwargs["quantization_config"] = quantization_config
        elif self.model_config["quantization"] == "auto":
            base_kwargs["torch_dtype"] = "auto"
            base_kwargs["device_map"] = "auto"
        else:
            base_kwargs["torch_dtype"] = torch_dtype

        logger.debug(f"base_kwargs: {base_kwargs}")
        model = AutoModelForCausalLM.from_pretrained(self.base_model, **base_kwargs)
        model.config.use_cache = self.model_config["use_cache"]
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

import bitsandbytes as bnb
import evaluate
import numpy as np
from peft import LoraConfig
import torch
from transformers.trainer_pt_utils import get_parameter_names
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


class CustomTrainer:
    def __init__(self, training_config, model, tokenizer, train_dataset, eval_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_config = training_config
        self.acc_metric = evaluate.load("accuracy")
        self.int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

    def train(self):
        trainer = self._setup_trainer()
        trainer.train()
        return trainer.model

    def _setup_trainer(self):
        # 데이터 콜레이터 설정
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=self.training_config["response_template"],
            tokenizer=self.tokenizer,
        )

        # LoRA 설정
        peft_config = LoraConfig(
            r=self.training_config["lora"]["r"],
            lora_alpha=self.training_config["lora"]["lora_alpha"],
            lora_dropout=self.training_config["lora"]["lora_dropout"],
            target_modules=self.training_config["lora"]["target_modules"],
            bias=self.training_config["lora"]["bias"],
            task_type=self.training_config["lora"]["task_type"],
        )

        # Optimizer 설정
        # https://huggingface.co/docs/transformers/en/perf_train_gpu_one#8-bit-adam
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.training_config["params"]["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        adam_bnb_optim = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            betas=(self.training_config["params"]["adam_beta1"], self.training_config["params"]["adam_beta2"]),
            eps=self.training_config["params"]["adam_epsilon"],
            lr=self.training_config["params"]["learning_rate"],
        )

        # SFT 설정
        sft_config = SFTConfig(
            do_train=self.training_config["params"]["do_train"],
            do_eval=self.training_config["params"]["do_eval"],
            lr_scheduler_type=self.training_config["params"]["lr_scheduler_type"],
            max_seq_length=self.training_config["params"]["max_seq_length"],
            per_device_train_batch_size=self.training_config["params"]["per_device_train_batch_size"],
            per_device_eval_batch_size=self.training_config["params"]["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.training_config["params"]["gradient_accumulation_steps"],
            gradient_checkpointing=self.training_config["params"]["gradient_checkpointing"],
            max_grad_norm=self.training_config["params"]["max_grad_norm"],
            num_train_epochs=self.training_config["params"]["num_train_epochs"],
            learning_rate=self.training_config["params"]["learning_rate"],
            weight_decay=self.training_config["params"]["weight_decay"],
            logging_strategy=self.training_config["params"]["logging_strategy"],
            save_strategy=self.training_config["params"]["save_strategy"],
            eval_strategy=self.training_config["params"]["eval_strategy"],
            logging_steps=self.training_config["params"]["logging_steps"],
            save_steps=self.training_config["params"]["save_steps"],
            eval_steps=self.training_config["params"]["eval_steps"],
            save_total_limit=self.training_config["params"]["save_total_limit"],
            save_only_model=self.training_config["params"]["save_only_model"],
            report_to=self.training_config["params"]["report_to"],
            run_name=self.training_config["params"]["run_name"],
            output_dir=self.training_config["params"]["output_dir"],
            overwrite_output_dir=self.training_config["params"]["overwrite_output_dir"],
        )

        return SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            args=sft_config,
            compute_metrics=self._compute_metrics,
            preprocess_logits_for_metrics=self._preprocess_logits_for_metrics,
            optimizers=(adam_bnb_optim, None),
        )

    # 모델의 logits를 조정하여 정답 토큰 부분만 출력하도록 설정
    def _preprocess_logits_for_metrics(self, logits, labels):
        logits = logits if not isinstance(logits, tuple) else logits[0]
        logit_idx = [
            self.tokenizer.vocab["1"],
            self.tokenizer.vocab["2"],
            self.tokenizer.vocab["3"],
            self.tokenizer.vocab["4"],
            self.tokenizer.vocab["5"],
        ]
        return logits[:, -2, logit_idx]  # -2: answer token, -1: eos token

    # metric 계산 함수
    def _compute_metrics(self, evaluation_result):
        logits, labels = evaluation_result
        # 토큰화된 레이블 디코딩
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = [x.split("<end_of_turn>")[0].strip() for x in labels]
        labels = [self.int_output_map[x] for x in labels]

        # softmax 함수를 사용하여 logits 변환
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
        predictions = np.argmax(probs, axis=-1)

        return self.acc_metric.compute(predictions=predictions, references=labels)

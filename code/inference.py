from loguru import logger
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class InferenceModel:
    def __init__(self, inference_config, model, tokenizer, test_dataset):
        self.inference_config = inference_config
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

    def run_inference(self):
        if not self.inference_config["do_test"]:
            logger.info("추론 단계를 생략합니다. inference do_test 설정을 확인하세요.")
            return

        results = self._inference(self.test_dataset)
        return self._save_results(results)

    def _inference(self, test_dataset):
        infer_results = []
        self.model.config.use_cache = True
        self.model.eval()

        with torch.inference_mode():
            for example in tqdm(test_dataset):
                outputs = self.model(
                    self.tokenizer.apply_chat_template(
                        example["messages"], tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    ).to("cuda")
                )

                logits = outputs.logits[:, -1].flatten().cpu()
                target_logits = [logits[self.tokenizer.vocab[str(i + 1)]] for i in range(5)]  # 선택지는 항상 5개
                probs = torch.nn.functional.softmax(torch.tensor(target_logits, dtype=torch.float32), dim=-1)
                predict_value = self.pred_choices_map[np.argmax(probs.detach().cpu().numpy())]

                infer_results.append({"id": example["id"], "answer": predict_value})

        return infer_results

    def _save_results(self, results):
        logger.info(self.inference_config["output_path"])
        pd.DataFrame(results).to_csv(self.inference_config["output_path"], index=False)

from ast import literal_eval

from loguru import logger
import numpy as np
import pandas as pd
from rag import BM25Retriever
import torch
from tqdm import tqdm


class InferenceModel:
    def __init__(self, data_config, inference_config, model, tokenizer):
        self.data_config = data_config
        self.inference_config = inference_config
        self.model = model
        self.tokenizer = tokenizer
        self.pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

    def run_inference(self):
        if not self.inference_config["do_test"]:
            return
        test_dataset = self._prepare_test_dataset()
        results = self._inference(test_dataset)
        return self._save_results(results)

    # TODO: config로 인자 분리
    def _retrieve(self, df):
        retriever = BM25Retriever(
            tokenize_fn=self.tokenizer.tokenize,
            doc_type="wikipedia",
            data_path="../data/",
            pickle_filename="wiki_mrc_bm25.pkl",
            doc_filename="wiki_mrc.json",
        )

        def combine_text(row):
            return row["paragraph"] + " " + row["problems"]["question"] + " " + " ".join(row["problems"]["choices"])

        queries = df.apply(combine_text, axis=1)
        top_k = 2
        retrive_result = retriever.bulk_retrieve(queries, top_k)
        return retrive_result["text"]

    def _prepare_test_dataset(self):
        test_df = pd.read_csv(self.data_config["test_path"])
        test_df["problems"] = test_df["problems"].apply(literal_eval)
        docs = self._retrieve(test_df)
        records = []
        for idx, row in test_df.iterrows():
            record = {
                "id": row["id"],
                "paragraph": row["paragraph"],
                "question": row["problems"]["question"],
                "choices": row["problems"]["choices"],
                "question_plus": row["problems"].get("question_plus", None),
                "doc": docs[idx],
            }
            # Include 'question_plus' if it exists
            if "question_plus" in row["problems"]:
                record["question_plus"] = row["problems"]["question_plus"]
            records.append(record)
        return pd.DataFrame(records)

    def _inference(self, test_df):
        infer_results = []
        self.model.config.use_cache = True
        self.model.eval()

        with torch.inference_mode():
            for _, row in tqdm(test_df.iterrows()):
                messages = self._create_messages(row)
                outputs = self.model(
                    self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    ).to("cuda")
                )

                logits = outputs.logits[:, -1].flatten().cpu()
                target_logits = [logits[self.tokenizer.vocab[str(i + 1)]] for i in range(len(row["choices"]))]
                probs = torch.nn.functional.softmax(torch.tensor(target_logits, dtype=torch.float32), dim=-1)
                predict_value = self.pred_choices_map[np.argmax(probs.detach().cpu().numpy())]

                infer_results.append({"id": row["id"], "answer": predict_value})

        return infer_results

    def _save_results(self, results):
        logger.info(self.inference_config["output_path"])
        pd.DataFrame(results).to_csv(self.inference_config["output_path"], index=False)

    def _create_messages(self, row):
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])

        # <보기>가 있을 때
        if row["question_plus"]:
            user_message = self.data_config["prompt"]["with_question"].format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                doc=row["doc"],
                choices=choices_string,
            )
        # <보기>가 없을 때
        else:
            user_message = self.data_config["prompt"]["no_question"].format(
                paragraph=row["paragraph"],
                question=row["question"],
                doc=row["doc"],
                choices=choices_string,
            )

        return [
            {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
            {"role": "user", "content": user_message},
        ]

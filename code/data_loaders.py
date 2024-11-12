from ast import literal_eval

from datasets import Dataset
from loguru import logger
import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, data_config):
        self.train_path = data_config["train_path"]
        self.test_path = data_config["test_path"]
        self.max_seq_length = data_config["max_seq_length"]
        self.test_size = data_config["test_size"]
        self.prompt_no_question = data_config["prompt"]["no_question"]
        self.prompt_with_question = data_config["prompt"]["with_question"]

    def prepare_datasets(self):
        df = self._load_data(self.train_path)
        dataset = Dataset.from_pandas(df)
        processed_dataset = self._process_dataset(dataset)
        tokenized_dataset = self._tokenize_dataset(processed_dataset)
        return self._split_dataset(tokenized_dataset)

    def _load_data(self, file_path):
        df = pd.read_csv(file_path)
        records = []
        for _, row in df.iterrows():
            problems = literal_eval(row["problems"])
            record = {
                "id": row["id"],
                "paragraph": row["paragraph"],
                "question": problems["question"],
                "choices": problems["choices"],
                "answer": problems.get("answer", None),
                "question_plus": problems.get("question_plus", None),
            }
            records.append(record)
        return pd.DataFrame(records)

    def _process_dataset(self, dataset):
        processed_dataset = []
        for i in range(len(dataset)):
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])])

            # <보기>가 있을 때
            if dataset[i]["question_plus"]:
                user_message = self.prompt_with_question.format(
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    question_plus=dataset[i]["question_plus"],
                    choices=choices_string,
                )
            # <보기>가 없을 때
            else:
                user_message = self.prompt_no_question.format(
                    paragraph=dataset[i]["paragraph"],
                    question=dataset[i]["question"],
                    choices=choices_string,
                )

            # chat message 형식으로 변환
            processed_dataset.append(
                {
                    "id": dataset[i]["id"],
                    "messages": [
                        {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": f"{dataset[i]['answer']}"},
                    ],
                    "label": dataset[i]["answer"],
                }
            )
        return Dataset.from_pandas(pd.DataFrame(processed_dataset))

    def _tokenize_dataset(self, dataset):
        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example["messages"])):
                output_texts.append(
                    self.tokenizer.apply_chat_template(
                        example["messages"][i],
                        tokenize=False,
                    )
                )
            return output_texts

        def tokenize(element):
            outputs = self.tokenizer(
                formatting_prompts_func(element),
                truncation=False,
                padding=False,
                return_overflowing_tokens=False,
                return_length=False,
            )
            return {
                "input_ids": outputs["input_ids"],
                "attention_mask": outputs["attention_mask"],
            }

        tokenized_dataset = dataset.map(
            tokenize,
            remove_columns=list(dataset.features),
            batched=True,
            num_proc=4,
            load_from_cache_file=True,
            desc="Tokenizing",
        )

        # 토큰 길이가 max_seq_length를 초과하는 데이터 필터링
        # 힌트: 1024보다 길이가 더 긴 데이터를 포함하면 더 높은 점수를 달성할 수 있을 것 같습니다!
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= self.max_seq_length)

        return tokenized_dataset

    def _split_dataset(self, dataset):
        split_dataset = dataset.train_test_split(test_size=self.test_size, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        logger.debug(self.tokenizer.decode(train_dataset[0]["input_ids"], skip_special_tokens=True))
        train_dataset_token_lengths = [len(train_dataset[i]["input_ids"]) for i in range(len(train_dataset))]
        logger.info(f"max token length: {max(train_dataset_token_lengths)}")
        logger.info(f"min token length: {min(train_dataset_token_lengths)}")
        logger.info(f"avg token length: {np.mean(train_dataset_token_lengths)}")

        return train_dataset, eval_dataset

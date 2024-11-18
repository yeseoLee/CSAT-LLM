from ast import literal_eval
from typing import Dict, List

from datasets import Dataset
from loguru import logger
import numpy as np
import pandas as pd
from rag import BM25Retriever


class DataLoader:
    def __init__(self, tokenizer, data_config):
        self.tokenizer = tokenizer
        self.retriever_config = data_config["retriever"]
        self.train_path = data_config["train_path"]
        self.test_path = data_config["test_path"]
        self.max_seq_length = data_config["max_seq_length"]
        self.test_size = data_config["test_size"]
        self.prompt_no_question = data_config["prompt"]["no_question"]
        self.prompt_with_question = data_config["prompt"]["with_question"]

    def prepare_datasets(self, is_train=True):
        """학습 또는 테스트용 데이터셋 준비"""
        if is_train:
            dataset = self._load_data(self.train_path)
            processed_dataset = self._process_dataset(dataset)
            tokenized_dataset = self._tokenize_dataset(processed_dataset)
            return self._split_dataset(tokenized_dataset)
        else:
            dataset = self._load_data(self.test_path)
            processed_dataset = self._process_dataset(dataset, is_train=False)
            return processed_dataset

    def _retrieve(self, df):
        retriever = BM25Retriever(
            tokenize_fn=self.tokenizer.tokenize,
            doc_type=self.retriever_config["doc_type"],
            data_path=self.retriever_config["data_path"],
            pickle_filename=self.retriever_config["pickle_filename"],
            doc_filename=self.retriever_config["doc_filename"],
        )

        def combine_text(row):
            return row["paragraph"] + " " + row["problems"]["question"] + " " + " ".join(row["problems"]["choices"])

        queries = df.apply(combine_text, axis=1)
        top_k = 2
        retrive_result = retriever.bulk_retrieve(queries, top_k)
        return retrive_result["text"]

    def _load_data(self, file_path) -> List[Dict]:
        """csv를 읽어오고 dictionary 배열 형태로 변환합니다."""
        df = pd.read_csv(file_path)
        df["problems"] = df["problems"].apply(literal_eval)
        docs = self._retrieve(df)

        records = []
        for idx, row in df.iterrows():
            problems = row["problems"]
            record = {
                "id": row["id"],
                "paragraph": row["paragraph"],
                "question": problems["question"],
                "choices": problems["choices"],
                "answer": problems.get("answer", None),
                "question_plus": problems.get("question_plus", None),
                "doc": docs[idx],
            }
            records.append(record)
        return records

    def _process_dataset(self, dataset: List[Dict], is_train=True):
        """데이터에 프롬프트 적용"""
        processed_data = []

        for _, row in dataset:
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])

            if row["question_plus"]:
                user_message = self.prompt_with_question.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    question_plus=row["question_plus"],
                    doc=row["doc"],
                    choices=choices_string,
                )
            else:
                user_message = self.prompt_no_question.format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    doc=row["doc"],
                    choices=choices_string,
                )

            messages = [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
            ]

            if is_train:
                messages.append({"role": "assistant", "content": f"{row['answer']}"})

            processed_data.append({"id": row["id"], "messages": messages, "label": row["answer"] if is_train else None})

        return Dataset.from_pandas(pd.DataFrame(processed_data))

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

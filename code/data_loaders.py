from ast import literal_eval
import os
from typing import Dict, List

from datasets import Dataset
from loguru import logger
import numpy as np
import pandas as pd
from rag import ElasticsearchRetriever


class DataLoader:
    def __init__(self, tokenizer, data_config):
        self.tokenizer = tokenizer
        self.retriever_config = data_config["retriever"]
        self.train_path = data_config["train_path"]
        self.test_path = data_config["test_path"]
        self.processed_train_path = data_config["processed_train_path"]
        self.processed_test_path = data_config["processed_test_path"]
        self.max_seq_length = data_config["max_seq_length"]
        self.test_size = data_config["test_size"]
        self.prompt_config = data_config["prompt"]

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
        if self.retriever_config["retriever_type"] == "Elasticsearch":
            retriever = ElasticsearchRetriever(
                index_name=self.retriever_config["index_name"],
            )
        elif self.retriever_config["retriever_type"] == "BM25":
            raise NotImplementedError("BM25는 더 이상 지원하지 않습니다. Elasticsearch를 사용해주세요...")
        else:
            return [""] * len(df)

        def _combine_text(row):
            if self.retriever_config["query_type"] == "pqc":
                return row["paragraph"] + " " + row["problems"]["question"] + " " + " ".join(row["problems"]["choices"])
            if self.retriever_config["query_type"] == "pq":
                return row["paragraph"] + " " + row["problems"]["question"]
            if self.retriever_config["query_type"] == "pc":
                return row["paragraph"] + " " + " ".join(row["problems"]["choices"])
            else:
                return row["paragraph"]

        top_k = self.retriever_config["top_k"]
        threshold = self.retriever_config["threshold"]
        query_max_length = self.retriever_config["query_max_length"]

        queries = df.apply(_combine_text, axis=1)
        filtered_queries = [(i, q) for i, q in enumerate(queries) if len(q) <= query_max_length]
        if not filtered_queries:
            return [""] * len(queries)

        indices, valid_queries = zip(*filtered_queries)
        retrieve_results = retriever.bulk_retrieve(valid_queries, top_k)
        # # [[{"text":"안녕하세요", "score":96}, {"text":"반갑습니다", "score":88},],]

        docs = [""] * len(queries)
        for idx, result in zip(indices, retrieve_results):
            docs[idx] = " ".join(item["text"] for item in result if item["score"] >= threshold)

        return docs

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
                "document": docs[idx],
            }
            records.append(record)
        logger.info("dataset 로드 및 retrive 완료.")
        return records

    def _process_dataset(self, dataset: List[Dict], is_train=True):
        """데이터에 프롬프트 적용"""

        # prompt 전처리된 데이터셋 파일이 존재한다면 이를 로드합니다.
        processed_df_path = self.processed_train_path if is_train else self.processed_test_path
        if os.path.isfile(processed_df_path):
            logger.info(f"전처리된 데이터셋을 불러옵니다: {processed_df_path}")
            processed_df = pd.read_csv(processed_df_path, encoding="utf-8")
            processed_df["messages"] = processed_df["messages"].apply(literal_eval)
            return Dataset.from_pandas(processed_df)

        # 데이터셋을 prompt 전처리하고 저장합니다.
        logger.info("데이터셋 전처리를 수행합니다.")
        processed_data = []
        for row in dataset:
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])

            # start
            if row["question_plus"]:
                message_start = self.prompt_config["start_with_plus"].format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    question_plus=row["question_plus"],
                    choices=choices_string,
                )
            else:
                message_start = self.prompt_config["start"].format(
                    paragraph=row["paragraph"],
                    question=row["question"],
                    choices=choices_string,
                )
            # mid
            if row["document"]:
                message_mid = self.prompt_config["mid_with_document"].format(
                    document=row["document"],
                )
            else:
                message_mid = self.prompt_config["start"]
            # end
            message_end = self.prompt_config["end"]

            user_message = message_start + message_mid + message_end
            messages = [
                {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                {"role": "user", "content": user_message},
            ]

            if is_train:
                messages.append({"role": "assistant", "content": f"{row['answer']}"})

            processed_data.append({"id": row["id"], "messages": messages, "label": row["answer"] if is_train else None})

        processed_df = pd.DataFrame(processed_data)
        logger.info("데이터셋 전처리가 완료되었습니다.")
        if processed_df_path:
            processed_df.to_csv(processed_df_path, index=False, encoding="utf-8")
            logger.info("전처리된 데이터셋이 저장되었습니다.")
        return Dataset.from_pandas(processed_df)

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
        logger.info(f"dataset length: {len(tokenized_dataset)}")
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= self.max_seq_length)
        logger.info(f"filtered dataset length: {len(tokenized_dataset)}")

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

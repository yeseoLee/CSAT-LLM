import os
import pickle
import sys


# rag 폴더 경로를 추가합니다.
sys.path.append(os.path.join(os.path.dirname(__file__), "rag"))

from ast import literal_eval
from typing import Dict, List

from datasets import Dataset
from loguru import logger
import numpy as np

# from kornia.data import Dataset
import pandas as pd
from rag import BM25Retriever
from rag.dpr_data import KorQuadDataset
from rag.encoder import KobertBiEncoder
from rag.indexers import DenseFlatIndexer
from rag.retriever import KorDPRRetriever, get_passage_file  # KorDPRRetriever 불러오기


class DataLoader:
    def __init__(self, tokenizer, data_config):
        self.tokenizer = tokenizer
        self.retriever_config = data_config["retriever"]
        self.train_path = data_config["train_path"]
        self.test_path = data_config["test_path"]
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
        logger.info("Starting _retrieve method")
        if self.retriever_config["retriever_type"] == "BM25":
            retriever = BM25Retriever(
                tokenize_fn=self.tokenizer.tokenize,
                doc_type=self.retriever_config["doc_type"],
                data_path=self.retriever_config["data_path"],
                pickle_filename=self.retriever_config["pickle_filename"],
                doc_filename=self.retriever_config["doc_filename"],
            )
        elif self.retriever_config["retriever_type"] == "DPR":
            # KorDPRRetriever 사용
            try:
                model = KobertBiEncoder()  # 모델 초기화
                model.load("./rag/output/my_model.pt")  # 모델 불러오기
                print("Model loaded successfully.")
                assert model is not None, "Model is None after loading."
            except Exception as e:
                print(f"Error while loading model: {e}")

            try:
                valid_dataset = KorQuadDataset("./rag/data/KorQuAD_v1.0_dev.json")  # 데이터셋 준비
                print("Valid dataset loaded successfully.")
            except Exception as e:
                print(f"Error while loading valid dataset: {e}")

            try:
                index = DenseFlatIndexer()  # 인덱스 준비
                index.deserialize(path="./rag/2050iter_flat/")
                print("Index loaded successfully.")
                assert index is not None, "Index is None after loading."
            except Exception as e:
                print(f"Error while loading index: {e}")

            ds_retriever = KorDPRRetriever(model=model, valid_dataset=valid_dataset, index=index)
            print("KorDPRRetriever initialized successfully.")

        def _combine_text(row):
            return row["paragraph"] + " " + row["problems"]["question"] + " " + " ".join(row["problems"]["choices"])

        top_k = self.retriever_config["top_k"]
        threshold = self.retriever_config["threshold"]
        queries = df.apply(_combine_text, axis=1)

        # BM25 또는 DPR을 사용하여 검색 결과 가져오기
        if self.retriever_config["retriever_type"] == "BM25":
            retrive_results = retriever.bulk_retrieve(queries, top_k)
            docs = []
            for result in retrive_results:
                docs.append(" ".join(item["text"] for item in result if item["score"] >= threshold))
        else:  # DPR인 경우
            docs = []
            for query in queries:
                passages = ds_retriever.retrieve(query=query, k=top_k)  # DPR으로 검색

                # passage 로딩 및 결합
                for idx, (passage, score) in enumerate(passages):
                    # passage ID에 해당하는 파일 경로 가져오기
                    path = get_passage_file([idx])
                    if path:
                        with open(path, "rb") as f:
                            passage_dict = pickle.load(f)
                            docs.append((passage_dict[idx], score))  # passage와 score 저장
                    else:
                        print(f"No passage found for ID: {idx}")

                    # 로깅 추가
                    logger.info(f"가연 Query: {query}")
                    logger.info(f"Rank {idx+1}: Score: {score:.4f}, Passage: {passage}")

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
        processed_data = []
        for row in dataset:
            choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])])

            # retriever에서 passage를 가져오기
            document = row.get("document", None)  # 이미 document가 있을 수 있음
            if not document:  # document가 없다면 retriever로부터 top-k passage를 검색
                query = row["paragraph"] + " " + row["problems"]["question"]
                docs = self._retrieve(pd.DataFrame([row]))[0]  # retrieve 결과 가져오기
                document = docs  # top-k passages가 document로 설정됨

                # 검색 결과 로깅
                logger.info(f"ID: {row['id']}")
                logger.info(f"Query: {query}")
                logger.info(f"Retrieved document: {document[:500]}...")

            # 메시지 시작 부분 작성
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

            # mid 부분에 retriever에서 가져온 passage를 추가
            message_mid = self.prompt_config["mid_with_document"].format(
                document=document,  # 이 부분에 문서 추가
            )

            message_end = self.prompt_config["end"]

            user_message = message_start + message_mid + message_end
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

from ast import literal_eval
import os
import pickle
from typing import Dict, List

from datasets import Dataset
from dotenv import load_dotenv
from loguru import logger
import numpy as np
import pandas as pd
from rag import ElasticsearchRetriever, Reranker
from rag.dpr_data import KorQuadDataset
from rag.encoder import KobertBiEncoder
from rag.indexers import DenseFlatIndexer
from rag.retriever import KorDPRRetriever, get_passage_file
from utils import load_config


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

    def prepare_datasets(self, is_train):
        """학습 또는 테스트용 데이터셋 준비"""
        # prompt 전처리된 데이터셋 파일이 존재한다면 이를 로드합니다.
        processed_df_path = self.processed_train_path if is_train else self.processed_test_path
        if os.path.isfile(processed_df_path):
            logger.info(f"전처리된 데이터셋을 불러옵니다: {processed_df_path}")
            processed_df = pd.read_csv(processed_df_path, encoding="utf-8")
            processed_df["messages"] = processed_df["messages"].apply(literal_eval)
            processed_dataset = Dataset.from_pandas(processed_df)
        else:
            dataset = self._load_data(is_train)
            processed_dataset = self._process_dataset(dataset, is_train)

        if is_train:
            tokenized_dataset = self._tokenize_dataset(processed_dataset)
            splitted_dataset = self._split_dataset(tokenized_dataset)
            return splitted_dataset
        return processed_dataset

    def _retrieve(self, df):  # noqa: C901
        if self.retriever_config["retriever_type"] == "Elasticsearch":
            retriever = ElasticsearchRetriever(
                index_name=self.retriever_config["index_name"],
            )
        elif self.retriever_config["retriever_type"] == "DPR":
            # KorDPRRetriever 사용
            try:
                model = KobertBiEncoder()  # 모델 초기화
                model.load("./rag/output/my_model.pt")  # 모델 불러오기
                logger.debug("Model loaded successfully.")
                assert model is not None, "Model is None after loading."
            except Exception as e:
                logger.debug(f"Error while loading model: {e}")

            try:
                valid_dataset = KorQuadDataset("./rag/data/KorQuAD_v1.0_dev.json")  # 데이터셋 준비
                logger.debug("Valid dataset loaded successfully.")
            except Exception as e:
                logger.debug(f"Error while loading valid dataset: {e}")

            try:
                index = DenseFlatIndexer()  # 인덱스 준비
                index.deserialize(path="./rag/2050iter_flat/")
                logger.debug("Index loaded successfully.")
                assert index is not None, "Index is None after loading."
            except Exception as e:
                logger.debug(f"Error while loading index: {e}")

            ds_retriever = KorDPRRetriever(model=model, valid_dataset=valid_dataset, index=index)
            logger.debug("KorDPRRetriever initialized successfully.")
        else:
            return [""] * len(df)

        def _combine_text(row):
            # NaN 값 처리
            paragraph = "" if pd.isna(row["paragraph"]) else str(row["paragraph"])
            if pd.isna(row["problems"]):
                problems = {"question": "", "choices": []}
            else:
                problems = row["problems"]
            question = str(problems.get("question", ""))
            choices = [str(choice) for choice in problems.get("choices", [])]

            if self.retriever_config["query_type"] == "pqc":
                return paragraph + " " + question + " " + " ".join(choices)
            if self.retriever_config["query_type"] == "pq":
                return paragraph + " " + question
            if self.retriever_config["query_type"] == "pc":
                return paragraph + " " + " ".join(choices)
            else:
                return paragraph

        top_k = self.retriever_config["top_k"]
        threshold = self.retriever_config["threshold"]
        query_max_length = self.retriever_config["query_max_length"]

        queries = df.apply(_combine_text, axis=1)
        if self.retriever_config["retriever_type"] == "Elasticsearch":
            filtered_queries = [(i, q) for i, q in enumerate(queries) if len(q) <= query_max_length]
            if not filtered_queries:
                return [""] * len(queries)

            indices, valid_queries = zip(*filtered_queries)
            retrieve_results = retriever.bulk_retrieve(valid_queries, top_k)
            rerank_k = self.retriever_config["rerank_k"]
            if rerank_k > 0:
                with Reranker() as reranker:
                    retrieve_results = reranker.rerank(valid_queries, retrieve_results, rerank_k)
            # [[{"text":"안녕하세요", "score":0.5}, {"text":"반갑습니다", "score":0.3},],]

            docs = [""] * len(queries)
            for idx, result in zip(indices, retrieve_results):
                docs[idx] = " ".join(item["text"] for item in result if item["score"] >= threshold)
                docs[idx] = docs[idx][: self.retriever_config["result_max_length"]]
        elif self.retriever_config["retriever_type"] == "DPR":  # DPR인 경우
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
                        logger.debug(f"No passage found for ID: {idx}")

                    # 로깅 추가
                    logger.info(f"가연 Query: {query}")
                    logger.info(f"Rank {idx+1}: Score: {score:.4f}, Passage: {passage}")

        return docs

    def _load_data(self, is_train) -> List[Dict]:
        """csv를 읽어오고 dictionary 배열 형태로 변환합니다."""
        file_path = self.train_path if is_train else self.test_path
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
                message_mid = self.prompt_config["mid"]
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
        processed_df_path = self.processed_train_path if is_train else self.processed_test_path
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


if __name__ == "__main__":
    config_folder = os.path.join(os.path.dirname(__file__), "..", "config/")
    load_dotenv(os.path.join(config_folder, ".env"))
    config = load_config()
    data_config = config["data"]

    def _retrieve(retriever_config, df):  # noqa: C901
        if retriever_config["retriever_type"] == "Elasticsearch":
            retriever = ElasticsearchRetriever(
                index_name=retriever_config["index_name"],
            )
        elif retriever_config["retriever_type"] == "BM25":
            raise NotImplementedError("BM25는 더 이상 지원하지 않습니다. Elasticsearch를 사용해주세요...")

        elif retriever_config["retriever_type"] == "DPR":
            # KorDPRRetriever 사용
            try:
                model = KobertBiEncoder()  # 모델 초기화
                model.load("./rag/output/my_model.pt")  # 모델 불러오기
                logger.debug("Model loaded successfully.")
                assert model is not None, "Model is None after loading."
            except Exception as e:
                logger.debug(f"Error while loading model: {e}")

            try:
                valid_dataset = KorQuadDataset("./rag/data/KorQuAD_v1.0_dev.json")  # 데이터셋 준비
                logger.debug("Valid dataset loaded successfully.")
            except Exception as e:
                logger.debug(f"Error while loading valid dataset: {e}")

            try:
                index = DenseFlatIndexer()  # 인덱스 준비
                index.deserialize(path="./rag/2050iter_flat/")
                logger.debug("Index loaded successfully.")
                assert index is not None, "Index is None after loading."
            except Exception as e:
                logger.debug(f"Error while loading index: {e}")

            ds_retriever = KorDPRRetriever(model=model, valid_dataset=valid_dataset, index=index)
            logger.debug("KorDPRRetriever initialized successfully.")

        else:
            return [""] * len(df)

        def _combine_text(row):
            if retriever_config["query_type"] == "pqc":
                return row["paragraph"] + " " + row["problems"]["question"] + " " + " ".join(row["problems"]["choices"])
            if retriever_config["query_type"] == "pq":
                return row["paragraph"] + " " + row["problems"]["question"]
            if retriever_config["query_type"] == "pc":
                return row["paragraph"] + " " + " ".join(row["problems"]["choices"])
            else:
                return row["paragraph"]

        top_k = retriever_config["top_k"]
        threshold = retriever_config["threshold"]
        query_max_length = retriever_config["query_max_length"]

        queries = df.apply(_combine_text, axis=1)
        if retriever_config["retriever_type"] == "Elasticsearch":
            filtered_queries = [(i, q) for i, q in enumerate(queries) if len(q) <= query_max_length]
            if not filtered_queries:
                return [""] * len(queries)

            indices, valid_queries = zip(*filtered_queries)
            retrieve_results = retriever.bulk_retrieve(valid_queries, top_k)
            rerank_k = retriever_config["rerank_k"]
            if rerank_k > 0:
                with Reranker() as reranker:
                    retrieve_results = reranker.rerank(valid_queries, retrieve_results, rerank_k)
            # [[{"text":"안녕하세요", "score":0.5}, {"text":"반갑습니다", "score":0.3},],]

            docs = [""] * len(queries)
            for idx, result in zip(indices, retrieve_results):
                docs[idx] = " ".join(
                    f"[{item['score']}]: {item['text']}" for item in result if item["score"] >= threshold
                )

        elif retriever_config["retriever_type"] == "DPR":  # DPR인 경우
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
                        logger.debug(f"No passage found for ID: {idx}")

                    # 로깅 추가
                    logger.info(f"Query: {query}")
                    logger.info(f"Rank {idx+1}: Score: {score:.4f}, Passage: {passage}")
        return docs

    def load_and_save(retriever_config, file_path) -> List[Dict]:
        """csv를 읽어오고 dictionary 배열 형태로 변환합니다."""
        df = pd.read_csv(file_path)
        df["problems"] = df["problems"].apply(literal_eval)
        docs = _retrieve(retriever_config, df)
        df["documents"] = docs
        df.to_csv(file_path.replace(".csv", "_retrieve.csv"), index=False)
        logger.debug("retrieve 결과가 csv로 저장되었습니다.")

    load_and_save(data_config["retriever"], data_config["train_path"])
    load_and_save(data_config["retriever"], data_config["test_path"])

import json
import os
import pickle
from typing import Dict, List, Optional

from datasets import load_dataset
from konlpy.tag import Okt
from loguru import logger
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer


class BM25Retriever:
    def __init__(
        self,
        tokenize_fn=None,
        doc_type: str = "wikipedia",
        data_path: Optional[str] = "../data/",
        pickle_filename: str = "wiki_bm25.pkl",
        doc_filename: Optional[str] = "wiki_document.json",
    ) -> None:
        self.tokenize_fn = tokenize_fn if tokenize_fn else lambda x: Okt().morphs(x)
        self.pickle_path = os.path.join(data_path, pickle_filename)

        self.bm25 = None
        self.corpus = []

        # 기존 인덱스 로드
        if os.path.exists(self.pickle_path):
            self._load_pickle()
            return

        # 데이터셋 로드 및 인덱스 생성
        self._load_dataset(doc_type, os.path.join(data_path, doc_filename))
        self._initialize_retriever(doc_type, os.path.join(data_path, doc_filename))

    def _load_dataset(self, doc_type, json_path):
        if doc_type == "wikipedia":
            logger.debug("위키피디아 데이터셋 로드")
            with open(json_path, "r", encoding="utf-8") as f:
                docs = json.load(f)
        elif doc_type == "namuwiki":
            logger.debug("나무위키 데이터셋 로드")
            dataset = load_dataset("heegyu/namuwiki-extracted")
            docs = dataset["train"]
        else:
            raise Exception(f"정의되지 않은 doc_type: {doc_type}")
        self.corpus = [f"{doc['title']}: {doc['text']}" for doc in docs]

    def _load_pickle(self):
        logger.debug("기존 BM25 인덱스 로드")
        with open(self.pickle_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.corpus = data["corpus"]

    def _initialize_retriever(self):
        logger.debug("새로운 BM25 인덱스 생성")

        tokenized_corpus = [self.tokenize_fn(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        with open(self.pickle_path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "corpus": self.corpus,
                },
                f,
            )
        logger.debug("인덱스 생성 완료")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        주어진 쿼리에 대해 상위 k개의 문서를 검색합니다.
        """
        if not self.bm25:
            raise Exception("BM25 모델이 초기화되지 않았습니다.")

        tokenized_query = self.tokenize_fn(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "text": self.corpus[idx],
                    "score": float(doc_scores[idx]),
                }
            )
        return results

    def bulk_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[Dict]]:
        """
        여러 쿼리에 대해 일괄적으로 검색을 수행합니다.
        """
        if not self.bm25:
            raise Exception("BM25 모델이 초기화되지 않았습니다.")

        results = []
        logger.debug(f"{len(queries)}개 쿼리 일괄 검색")
        # 모든 쿼리를 한 번에 토크나이징
        tokenized_queries = [self.tokenize_fn(query) for query in queries]

        # 각 쿼리별로 검색 수행
        for tokenized_query in tokenized_queries:
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(doc_scores)[-top_k:][::-1]

            query_results = []
            for idx in top_indices:
                query_results.append(
                    {
                        "text": self.corpus[idx],
                        "score": float(doc_scores[idx]),
                    }
                )
            results.append(query_results)

        return results


if __name__ == "__main__":
    os.chdir("..")
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    retriever = BM25Retriever(
        tokenize_fn=tokenizer.tokenize,
        doc_type="wikipedia",
        data_path="../data/",
        pickle_filename="wiki_bm25.pkl",
        doc_filename="wiki_dump.json",
    )

    query = "선비들 수만 명이 대궐 앞에 모여 만 동묘와 서원을 다시 설립할 것을 청하니, (가)이/가 크게 노하여 한성부의 조례(皂隷)와 병졸로 하여 금 한 강 밖으로 몰아내게 하고 드디어 천여 곳의 서원을 철폐하고 그 토지를 몰수하여 관에 속하게 하였다.－대한계년사"  # noqa: E501
    results = retriever.retrieve(query, top_k=5)

    for i, result in enumerate(results, 1):
        logger.debug(f"\n검색 결과 {i}")
        logger.debug(f"점수: {result['score']:.4f}")
        logger.debug(f"내용: {result['text'][:200]}...")

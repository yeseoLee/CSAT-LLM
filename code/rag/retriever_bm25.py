import json
import os
import pickle
from typing import Dict, List, Optional

from loguru import logger
import numpy as np
from rank_bm25 import BM25Okapi


# from konlpy.tag import Okt
# okt = Okt()
# def okt_specific_pos_tokenizer(text, stem=True, norm=True):
#     # pos 태깅 수행
#     pos_tagged = okt.pos(text, stem=stem, norm=norm)
#     # 명사(Noun), 형용사(Adjective), 동사(Verb)만 필터링
#     filtered_words = [word for word, pos in pos_tagged if pos in ["Noun", "Adjective", "Verb"]]
#     return filtered_words


# Deprecated: 너무 느려서 더 이상 사용하지 않습니다.
class BM25Retriever:
    def __init__(
        self,
        tokenize_fn=None,
        data_path: Optional[str] = "../data/",
        pickle_filename: str = "wiki_bm25.pkl",
        doc_filename: Optional[str] = "wiki_document.json",
    ) -> None:
        self.tokenize_fn = tokenize_fn if tokenize_fn else lambda x: x.split()
        self.pickle_path = os.path.join(data_path, pickle_filename)
        self.bm25 = None
        self.corpus = []

        # 데이터셋 로드
        self._load_dataset(os.path.join(data_path, doc_filename))

        # 기존 인덱스 로드
        if os.path.exists(self.pickle_path):
            self._load_pickle()
            return

        # 인덱스 생성
        self._initialize_retriever()

    def _load_dataset(self, json_path):
        logger.info("문서 데이터셋 로드")
        with open(json_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        self.corpus = [f"{doc['title']}: {doc['text']}" for doc in docs]

    def _load_pickle(self):
        logger.info("기존 BM25 인덱스 로드")
        with open(self.pickle_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]

    def _initialize_retriever(self):
        logger.info("새로운 BM25 인덱스 생성")

        tokenized_corpus = [self.tokenize_fn(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        with open(self.pickle_path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                },
                f,
            )
        logger.info("인덱스 생성 완료")

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
        logger.info(f"{len(queries)}개 쿼리 일괄 검색")
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

        logger.info(f"{len(queries)}개 쿼리 일괄 검색 완료")
        return results


if __name__ == "__main__":
    os.chdir("..")
    retriever = BM25Retriever(
        tokenize_fn=None,
        data_path="../data/",
        pickle_filename="wiki_bm25.pkl",
        doc_filename="wiki.json",
    )

    query = "선비들 수만 명이 대궐 앞에 모여 만 동묘와 서원을 다시 설립할 것을 청하니, (가)이/가 크게 노하여 한성부의 조례(皂隷)와 병졸로 하여 금 한 강 밖으로 몰아내게 하고 드디어 천여 곳의 서원을 철폐하고 그 토지를 몰수하여 관에 속하게 하였다.－대한계년사"  # noqa: E501
    results = retriever.retrieve(query, top_k=5)

    for i, result in enumerate(results, 1):
        logger.debug(f"\n검색 결과 {i}")
        logger.debug(f"점수: {result['score']:.4f}")
        logger.debug(f"내용: {result['text'][:200]}...")

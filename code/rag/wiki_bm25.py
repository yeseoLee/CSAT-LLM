from contextlib import contextmanager
import json
import os
import pickle
import time
from typing import Dict, List, Optional

from datasets import load_dataset
from konlpy.tag import Okt
from loguru import logger
import numpy as np
from rank_bm25 import BM25Okapi


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.debug(f"[{name}] done in {time.time() - t0:.3f} s")


class WikiBM25Retriever:
    def __init__(
        self,
        tokenize_fn=None,
        wiki_type: str = "wikipedia",
        data_path: Optional[str] = "../data/",
        pickle_filename: str = "wiki_bm25.pkl",
        wiki_filename: Optional[str] = "wiki_document.json",
    ) -> None:
        self.tokenize_fn = tokenize_fn if tokenize_fn else lambda x: Okt().morphs(x)
        self.pickle_path = os.path.join(data_path, pickle_filename)

        self.bm25 = None
        self.titles = []
        self.contents = []

        if os.path.exists(self.pickle_path):
            self._load_pickle()
            return
        self._initialize_retriever(wiki_type, os.path.join(data_path, wiki_filename))

    def _load_pickle(self):
        with timer("기존 BM25 인덱스 로드"):
            with open(self.pickle_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.titles = data["titles"]
                self.contents = data["contents"]

    def _initialize_retriever(self, wiki_type, json_path):
        with timer("새로운 BM25 인덱스 생성"):
            if wiki_type == "wikipedia":
                # 위키피디아 데이터셋 로드
                logger.debug("위키피디아 데이터셋을 불러옵니다...")
                with open(json_path, "r", encoding="utf-8") as f:
                    wiki_docs = json.load(f)
            elif wiki_type == "namuwiki":
                # 나무위키 데이터셋 로드
                logger.debug("나무위키 데이터셋을 불러옵니다...")
                dataset = load_dataset("heegyu/namuwiki-extracted")
                wiki_docs = dataset["train"]
            else:
                raise Exception(f"정의되지 않은 wiki_type: {wiki_type}")

            self.titles = [doc["title"] for doc in wiki_docs]
            self.contents = [doc["text"] for doc in wiki_docs]

            # BM25 모델 생성
            logger.debug("BM25 모델을 생성합니다...")
            corpus = [f"{title}: {content}" for title, content in zip(self.titles, self.contents)]
            tokenized_corpus = [self.tokenize_fn(doc) for doc in corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)

            with open(self.pickle_path, "wb") as f:
                pickle.dump(
                    {
                        "bm25": self.bm25,
                        "titles": self.titles,
                        "contents": self.contents,
                    },
                    f,
                )
            logger.debug("인덱스 생성이 완료되었습니다.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        주어진 쿼리에 대해 상위 k개의 문서를 검색합니다.

        Args:
            query (str): 검색 쿼리
            top_k (int): 반환할 문서 개수

        Returns:
            List[Dict]: 검색 결과 리스트. 각 결과는 title, content, contributors, score를 포함
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
                    "title": self.titles[idx],
                    "content": self.contents[idx],
                    "score": float(doc_scores[idx]),
                }
            )
        return results

    def bulk_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[Dict]]:
        """
        여러 쿼리에 대해 일괄적으로 검색을 수행합니다.

        Args:
            queries (List[str]): 검색할 쿼리 리스트
            top_k (int): 각 쿼리당 반환할 문서 개수

        Returns:
            List[List[Dict]]: 각 쿼리별 검색 결과 리스트
        """
        if not self.bm25:
            raise Exception("BM25 모델이 초기화되지 않았습니다.")

        results = []
        with timer(f"{len(queries)}개 쿼리 일괄 검색"):
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
                            "title": self.titles[idx],
                            "content": self.contents[idx],
                            "score": float(doc_scores[idx]),
                        }
                    )
                results.append(query_results)

        return results


if __name__ == "__main__":
    os.chdir("..")
    retriever = WikiBM25Retriever(
        tokenize_fn=None,
        wiki_type="namuwiki",
        data_path="../data/",
        pickle_filename="namu_bm25.pkl",
        wiki_filename="",
    )

    query = "세계수의 미궁"
    results = retriever.retrieve(query, top_k=5)

    for i, result in enumerate(results, 1):
        logger.debug(f"\n검색 결과 {i}")
        logger.debug(f"제목: {result['title']}")
        logger.debug(f"점수: {result['score']:.4f}")
        logger.debug(f"내용: {result['content'][:200]}...")

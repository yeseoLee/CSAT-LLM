import json
import os
from typing import Dict, List, Optional

import bm25s
from bm25s.tokenization import Tokenizer
from datasets import load_dataset
from konlpy.tag import Okt
from loguru import logger


class WikiBM25sRetriever:
    def __init__(
        self,
        tokenize_fn=None,
        wiki_type: str = "wikipedia",
        data_path: Optional[str] = "../data/",
        save_name: str = "wiki_bm25s",
        wiki_filename: Optional[str] = "wiki_document.json",
    ) -> None:
        tokenize_fn = tokenize_fn if tokenize_fn else lambda x: Okt().morphs(x)
        self.save_dir = os.path.join(data_path, save_name)
        os.makedirs(data_path, exist_ok=True)

        # 토크나이저 초기화
        self.tokenizer = Tokenizer(stemmer=None, stopwords=[], splitter=tokenize_fn)

        self.bm25 = None
        self.titles = []
        self.contents = []

        # 기존 인덱스가 있으면 로드, 없으면 새로 생성
        if os.path.exists(self.save_dir):
            self._load_index()
        else:
            self._initialize_retriever(wiki_type, os.path.join(data_path, wiki_filename))

    def _load_index(self):
        logger.debug("기존 BM25 인덱스 로드")
        self.bm25 = bm25s.BM25.load(self.save_dir, load_corpus=True)
        self.tokenizer.load_vocab(self.save_dir)

    def _initialize_retriever(self, wiki_type, json_path):
        logger.debug("새로운 BM25 인덱스 생성")
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
        corpus_tokens = self.tokenizer.tokenize(corpus)
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens)
        self.bm25.save(self.save_dir, corpus=corpus)
        self.tokenizer.save_vocab(self.save_dir)
        logger.debug("인덱스 생성이 완료되었습니다.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        주어진 쿼리에 대해 상위 k개의 문서를 검색합니다.
        """
        if not self.bm25:
            raise Exception("BM25 모델이 초기화되지 않았습니다.")

        query_tokens = self.tokenizer.tokenize(query)
        results, scores = self.bm25.retrieve(query_tokens, k=top_k)

        formatted_results = []
        for idx, score in zip(results[0], scores[0]):
            formatted_results.append({"title": self.titles[idx], "content": self.contents[idx], "score": float(score)})
        return formatted_results

    def bulk_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[Dict]]:
        """
        여러 쿼리에 대해 일괄적으로 검색을 수행합니다.
        """
        if not self.bm25:
            raise Exception("BM25 모델이 초기화되지 않았습니다.")

        logger.debug(f"{len(queries)}개 쿼리 일괄 검색")
        # 쿼리 토크나이징
        query_tokens = [self.tokenizer.tokenize(query) for query in queries]

        # 일괄 검색 수행
        results, scores = self.bm25.retrieve(query_tokens, k=top_k)

        # 결과 포매팅
        formatted_results = []
        for query_results, query_scores in zip(results, scores):
            query_formatted = []
            for idx, score in zip(query_results, query_scores):
                query_formatted.append(
                    {"title": self.titles[idx], "content": self.contents[idx], "score": float(score)}
                )
            formatted_results.append(query_formatted)
        return formatted_results


if __name__ == "__main__":
    os.chdir("..")
    retriever = WikiBM25sRetriever(
        wiki_type="wikipedia",
        data_path="../data/",
        save_name="wiki_mrc_bm25s",
        wiki_filename="wiki_mrc.json",
    )

    # 단일 쿼리 테스트
    query = "세계수의 미궁"
    results = retriever.retrieve(query, top_k=3)

    logger.debug("\n단일 쿼리 검색 결과:")
    for i, result in enumerate(results, 1):
        logger.debug(f"\n검색 결과 {i}")
        logger.debug(f"제목: {result['title']}")
        logger.debug(f"점수: {result['score']:.4f}")
        logger.debug(f"내용: {result['content'][:200]}...")

    # 다중 쿼리 테스트
    queries = ["세계수의 미궁", "인공지능", "대한민국"]
    bulk_results = retriever.bulk_retrieve(queries, top_k=3)

    logger.debug("\n다중 쿼리 검색 결과:")
    for query_idx, query_results in enumerate(bulk_results):
        logger.debug(f"\n쿼리: {queries[query_idx]}")
        for i, result in enumerate(query_results, 1):
            logger.debug(f"\n검색 결과 {i}")
            logger.debug(f"제목: {result['title']}")
            logger.debug(f"점수: {result['score']:.4f}")
            logger.debug(f"내용: {result['content'][:200]}...")

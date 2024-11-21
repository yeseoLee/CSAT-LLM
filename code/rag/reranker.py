import gc
import os
from typing import Dict, List

from dotenv import load_dotenv
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .retriever_elastic import ElasticsearchRetriever


class Reranker:
    def __init__(
        self,
        model_path: str = "Dongjin-kr/ko-reranker",
        batch_size: int = 128,
        max_length: int = 512,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.batch_size = batch_size
        self.max_length = max_length

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # GPU 메모리 정리
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    def _exp_normalize(self, x):
        y = np.exp(x - x.max(axis=1, keepdims=True))
        return y / y.sum(axis=1, keepdims=True)

    def rerank(self, queries: List[str], retrieve_results: List[List[Dict]], topk: int = 5) -> List[List[Dict]]:
        # 입력 데이터 준비
        all_pairs = []
        for query, results in zip(queries, retrieve_results):
            for result in results:
                all_pairs.append([query, result["text"]])

        # 배치 처리
        all_scores = []
        for i in tqdm(range(0, len(all_pairs), self.batch_size), desc="Reranking"):
            batch_pairs = all_pairs[i : i + self.batch_size]

            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                batch_scores = self.model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()
                all_scores.extend(batch_scores)

        all_scores = np.array(all_scores)

        reranked_results = []
        start = 0
        for results in retrieve_results:
            end = start + len(results)
            scores = all_scores[start:end]
            scores = self._exp_normalize(scores.reshape(1, -1)).flatten()
            top_indices = np.argsort(scores)[-topk:][::-1]

            reranked_batch = [{"text": results[i]["text"], "score": float(scores[i])} for i in top_indices]
            reranked_results.append(reranked_batch)
            start = end

        return reranked_results


if __name__ == "__main__":
    config_folder = os.path.join(os.path.dirname(__file__), "..", "..", "config")
    load_dotenv(os.path.join(config_folder, ".env"))

    reranker = Reranker(
        model_path="Dongjin-kr/ko-reranker",
        batch_size=128,
        max_length=512,
    )
    retriever = ElasticsearchRetriever(
        index_name="two-wiki-index",
    )

    query = "선비들 수만 명이 대궐 앞에 모여 만 동묘와 서원을 다시 설립할 것을 청하니, (가)이/가 크게 노하여 한성부의 조례(皂隷)와 병졸로 하여 금 한 강 밖으로 몰아내게 하고 드디어 천여 곳의 서원을 철폐하고 그 토지를 몰수하여 관에 속하게 하였다.－대한계년사"  # noqa: E501
    retriever_result = retriever.retrieve(query, top_k=5)
    logger.debug("Elasticsearch Retriever")
    logger.debug(f"{retriever_result[:5]}")

    reranked_results = reranker.rerank(queries=[query], retrieve_results=[retriever_result], topk=3)
    logger.debug("Reranker")
    logger.debug(f"{reranked_results}")

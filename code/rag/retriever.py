from collections import defaultdict
from glob import glob
import math
import os
import pickle
import typing

# from utils import get_passage_file
from typing import List

from dpr_data import KorQuadDataset, KorQuadSampler, korquad_collator
from encoder import KobertBiEncoder
from indexers import DenseFlatIndexer
from loguru import logger
import torch
from torch import tensor as T
from tqdm import tqdm


def get_wiki_filepath(data_dir):
    return glob(f"{data_dir}/*/wiki_*")


def wiki_worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # logger.debug(dataset)
    # dataset =
    overall_start = dataset.start
    overall_end = dataset.end
    split_size = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    # end_idx = min((worker_id+1) * split_size, len(dataset.data))
    dataset.start = overall_start + worker_id * split_size
    dataset.end = min(dataset.start + split_size, overall_end)  # index error 방지


def get_passage_file(p_id_list: typing.List[int]) -> str:
    """passage id를 받아서 해당되는 파일 이름을 반환합니다."""
    target_file = None
    p_id_max = max(p_id_list)
    p_id_min = min(p_id_list)

    # 현재 파일의 경로를 기준으로 'processed_passages' 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    passages_dir = os.path.join(current_dir, "processed_passages")

    # 'processed_passages' 디렉터리에서 파일을 찾음
    for f in glob(f"{passages_dir}/*.p"):
        file_name = os.path.basename(f)
        s, e = file_name.split(".")[0].split("-")
        s, e = int(s), int(e)

        if p_id_min >= s and p_id_max <= e:
            target_file = f
            break

    if target_file is None:
        logger.debug(f"No file found for passage IDs: {p_id_list}")

    return target_file


class KorDPRRetriever:
    def __init__(self, model, valid_dataset, index, val_batch_size: int = 64, device="cuda:0"):
        # 모델이 경로로 주어진 경우 로드
        if isinstance(model, str):
            self.model = KobertBiEncoder()
            self.model.load(model)
        else:
            self.model = model

        # 모델을 해당 디바이스로 이동
        self.model = self.model.to(device)
        self.model.eval()

        # 데이터셋 로드
        self.valid_dataset = valid_dataset

        # 인덱스가 경로로 주어진 경우 로드
        if isinstance(index, str):
            self.index = DenseFlatIndexer()
            self.index.deserialize(path=index)
        else:
            self.index = index
        self.model = model.to(device)
        self.device = device
        self.tokenizer = valid_dataset.tokenizer
        self.val_batch_size = val_batch_size
        self.valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset.dataset,
            batch_sampler=KorQuadSampler(valid_dataset.dataset, batch_size=val_batch_size, drop_last=False),
            collate_fn=lambda x: korquad_collator(x, padding_value=valid_dataset.pad_token_id),
            num_workers=4,
        )
        self.index = index

    def val_top_k_acc(self, k: List[int] = [5] + list(range(10, 101, 10))):
        """validation set에서 top k 정확도를 계산합니다."""

        self.model.eval()  # 평가 모드
        k_max = max(k)
        sample_cnt = 0
        retr_cnt = defaultdict(int)
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc="valid"):
                # batch_q, batch_q_attn_mask, batch_p_id, batch_p, batch_p_attn_mask
                q, q_mask, p_id, a, a_mask = batch
                q, q_mask = (
                    q.to(self.device),
                    q_mask.to(self.device),
                )
                q_emb = self.model(q, q_mask, "query")  # bsz x bert_dim
                result = self.index.search_knn(query_vectors=q_emb.cpu().numpy(), top_docs=k_max)

                for (pred_idx_lst, _), true_idx, _a, _a_mask in zip(result, p_id, a, a_mask):
                    a_len = _a_mask.sum()
                    _a = _a[:a_len]
                    _a = _a[1:-1]
                    _a_txt = self.tokenizer.decode(_a).strip()
                    docs = [pickle.load(open(get_passage_file([idx]), "rb"))[idx] for idx in pred_idx_lst]

                    for _k in k:
                        if _a_txt in " ".join(docs[:_k]):
                            retr_cnt[_k] += 1

                bsz = q.size(0)
                sample_cnt += bsz
        retr_acc = {_k: float(v) / float(sample_cnt) for _k, v in retr_cnt.items()}
        return retr_acc

    def retrieve(self, query: str, k: int = 100):
        """주어진 쿼리에 대해 가장 유사도가 높은 passage를 반환합니다."""
        self.model.eval()  # 평가 모드
        tok = self.tokenizer.batch_encode_plus([query], truncation=True, padding=True, max_length=512)

        # Ensure tensors are moved to the same device as the model (cuda:0)
        input_ids = T(tok["input_ids"]).to(self.device)
        attention_mask = T(tok["attention_mask"]).to(self.device)

        with torch.no_grad():
            out = self.model(input_ids, attention_mask, "query")

        result = self.index.search_knn(query_vectors=out.cpu().numpy(), top_docs=k)
        # logger.debug(result)
        # 원문 가져오기
        passages = []
        for idx, sim in zip(*result[0]):
            # logger.debug(idx)
            path = get_passage_file([idx])
            if not path:
                logger.debug(f"올바른 경로에 피클화된 위키피디아가 있는지 확인하세요.No single passage path for {idx}")
                continue
            with open(path, "rb") as f:
                passage_dict = pickle.load(f)
            # logger.debug(f"passage : {passage_dict[idx]}, sim : {sim}")
            passages.append((passage_dict[idx], sim))
            # logger.debug("성공!!!!!!")
        return passages


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--query", "-q", type=str, required=True)
    # parser.add_argument("--k", "-k", type=int, required=True)
    # args = parser.parse_args()

    model = KobertBiEncoder()
    model.load("./output/my_model.pt")
    model.eval()
    valid_dataset = KorQuadDataset("./data/KorQuAD_v1.0_dev.json")
    index = DenseFlatIndexer()
    index.deserialize(path="./2050iter_flat")

    retriever = KorDPRRetriever(model=model, valid_dataset=valid_dataset, index=index)

    # 'query'와 'k' 값을 설정합니다.
    query = "(가)이/가 크게 노하여 한성부의 조례(皂隷)와 병졸로 하여 금 한 강 밖으로 몰아내게 하고 드디어 천여 곳의 서원을 철폐하고 그 토지를 몰수하여 관에 속하게 하였다.－대한계년사 －"  # noqa: E501
    # query = "성학집요의 저자는?"
    k = 10  # 상위 20개 유사한 passage를 출력하려면 k를 20으로 설정

    # retrieve 메서드를 호출하여 가장 유사도가 높은 k개의 passage를 찾습니다.
    passages = retriever.retrieve(query=query, k=k)

    # 출력: 유사도 높은 passage와 그 유사도를 출력합니다.
    for idx, (passage, sim) in enumerate(passages):
        logger.debug(f"Rank {idx + 1} | Similarity: {sim:.4f} | Passage: {passage}")

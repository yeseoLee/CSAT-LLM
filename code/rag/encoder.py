from copy import deepcopy
import logging
import os

import torch
from transformers import BertModel


# 두 개의 BertModel을 사용하여 passage와 query를 encoding을 실행
# 토크나이징 후에 토큰을 고정된 크기의 벡터로 변경

# 로그 디렉토리 생성 (없으면 새로 생성)
os.makedirs("logs", exist_ok=True)

# 로깅 설정: 로그를 파일로 저장하고 디버깅 레벨로 설정
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()


# KobertBiEncoder 클래스 정의
class KobertBiEncoder(torch.nn.Module):
    def __init__(self):
        # torch.nn.Module의 초기화 함수 호출
        super(KobertBiEncoder, self).__init__()
        # passage(문서)를 처리하는 BERT 모델
        self.passage_encoder = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)
        # query(질의)를 처리하는 BERT 모델
        self.query_encoder = BertModel.from_pretrained("monologg/kobert", trust_remote_code=True)
        # BERT 모델의 pooler output(임베딩 크기) 설정
        self.emb_sz = self.passage_encoder.pooler.dense.out_features  # get cls token dim

    def forward(self, x: torch.LongTensor, attn_mask: torch.LongTensor, type: str = "passage") -> torch.FloatTensor:
        """passage 또는 query를 BERT로 인코딩합니다."""
        # type이 'passage' 또는 'query'인지 확인
        assert type in (
            "passage",
            "query",
        ), "type should be either 'passage' or 'query'"
        # type에 따라 다른 인코더 사용
        if type == "passage":
            # 문서(passage) 인코딩
            return self.passage_encoder(input_ids=x, attention_mask=attn_mask).pooler_output
        else:
            # 질의(query) 인코딩
            return self.query_encoder(input_ids=x, attention_mask=attn_mask).pooler_output

    def checkpoint(self, model_ckpt_path):
        # 모델의 가중치를 파일로 저장
        torch.save(deepcopy(self.state_dict()), model_ckpt_path)
        logger.debug(f"model self.state_dict saved to {model_ckpt_path}")

    def load(self, model_ckpt_path):
        # 저장된 가중치를 파일에서 로드
        with open(model_ckpt_path, "rb") as f:
            state_dict = torch.load(f)
        # 모델에 로드된 가중치 적용
        self.load_state_dict(state_dict)
        logger.debug(f"model self.state_dict loaded from {model_ckpt_path}")

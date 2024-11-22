import logging
import os

from chunk_data import save_orig_passage, save_title_index_map
from dpr_data import KorQuadDataset
from encoder import KobertBiEncoder

# 외부 스크립트에서 IndexRunner 임포트
from index_runner import IndexRunner
from indexers import DenseFlatIndexer  # index 관련
from retriever import KorDPRRetriever  # retriever.py에서 가져오기
import torch
from trainer import Trainer
import transformers


transformers.logging.set_verbosity_error()  # 토크나이저 초기화 관련 경고 억제

# 로깅 설정
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()


# 모델 존재 여부 확인 함수
def check_if_model_exists(model_path: str):
    """모델 체크포인트가 존재하는지 확인하는 함수"""
    return os.path.exists(model_path)


# 위키 데이터 처리 함수
def process_wiki_data():
    """
    위키 데이터를 처리하고 필요한 피클 파일을 생성합니다.
    """
    processed_passages_path = "processed_passages"

    if not os.path.exists(processed_passages_path):
        logger.debug(f"'{processed_passages_path}' 폴더가 존재하지 않습니다. 데이터 처리를 시작합니다.")

        # 1. chunk 데이터를 피클 파일로 변환하여 processed_passages 폴더에 저장 (약 10분 소요)
        save_orig_passage()

        # 2. 제목과 인덱스 매핑 저장
        save_title_index_map()

        logger.debug("데이터 처리가 완료되었습니다.")
    else:
        logger.debug(f"'{processed_passages_path}' 폴더가 이미 존재합니다. 데이터 처리를 건너뜁니다.")


if __name__ == "__main__":
    # 위키 데이터 처리
    # processed_passage 폴더 내에 피클화된 데이터가 저장됩니다. 10분 소요
    process_wiki_data()

    # 모델 경로 설정
    model_path = "./output/my_model.pt"

    # korquad 데이터로 모델을 학습시켜줍니다.
    # 모델이 이미 존재하면 학습을 건너뜁니다
    if check_if_model_exists(model_path):
        logger.debug(f"이미 학습된 모델이 {model_path}에 존재합니다. 학습을 건너뜁니다.")
    else:
        logger.debug("학습된 모델이 없습니다. 학습을 시작합니다.")

        # 모델과 데이터셋 준비
        device = torch.device("cuda:0")
        model = KobertBiEncoder()
        train_dataset = KorQuadDataset("./data/KorQuAD_v1.0_train.json")
        valid_dataset = KorQuadDataset("./data/KorQuAD_v1.0_dev.json")

        # Trainer 객체 생성 및 학습 시작
        my_trainer = Trainer(
            model=model,
            device=device,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            num_epoch=10,
            batch_size=128 - 32,
            lr=1e-5,
            betas=(0.9, 0.99),
            num_warmup_steps=1000,
            num_training_steps=100000,
            valid_every=30,
            best_val_ckpt_path=model_path,
        )

        # 학습 상태 불러오기
        # my_trainer.load_training_state()

        # 학습 시작
        my_trainer.fit()

    # Indexing을 실행하는 코드 (IndexRunner 사용)
    index_path = "./2050iter_flat"  # 인덱스 파일 경로 설정

    # 인덱스가 이미 존재하면 인덱싱을 건너뜁니다
    if not os.path.exists(index_path):
        logger.info("인덱스가 존재하지 않습니다. 인덱싱을 시작합니다.")
        index_runner = IndexRunner(
            data_dir="./text",
            model_ckpt_path="./output/my_model.pt",
            index_output=index_path,
        )
        index_runner.run()
    else:
        logger.info(f"인덱스 파일 '{index_path}'가 이미 존재합니다. 인덱싱을 건너뜁니다.")

    # index 파일 로딩
    index = DenseFlatIndexer()
    index.deserialize(path=index_path)  # 이미 생성된 인덱스 파일을 로드

    # retriever.py로부터 KorDPRRetriever 객체를 생성하여 쿼리 실행
    model = KobertBiEncoder()
    model.load("./output/my_model.pt")
    model.eval()

    valid_dataset = KorQuadDataset("./data/KorQuAD_v1.0_dev.json")
    retriever = KorDPRRetriever(model=model, valid_dataset=valid_dataset, index=index)

    # 'query'와 'k' 값을 설정합니다.
    query = "중국의 천안문 사태가 일어난 년도는?"
    k = 10  # 상위 10개 유사한 passage를 출력

    # retrieve 메서드를 호출하여 가장 유사도가 높은 k개의 passage를 찾습니다.
    passages = retriever.retrieve(query=query, k=k)

    # 출력: 유사도 높은 passage와 그 유사도를 출력합니다.
    for idx, (passage, sim) in enumerate(passages):
        logger.debug(f"Rank {idx + 1} | Similarity: {sim:.4f} | Passage: {passage}")

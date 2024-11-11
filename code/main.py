from .data_loaders import DataLoader
from .inference import Inferencer
from .model import ModelHandler
from .trainer import CustomTrainer
from .util import load_config, set_logger, set_seed


def main():
    # 설정 로드
    config = load_config()

    # 로그 설정
    set_logger(log_file=config["log"]["file"], log_level=config["log"]["level"])

    # 시드 설정
    set_seed()

    # 모델 및 토크나이저 설정
    model_handler = ModelHandler(config["model"])
    model, tokenizer = model_handler.setup()

    # 데이터 처리
    data_processor = DataLoader(config["data"])
    data_processor.tokenizer = tokenizer  # 토크나이저 설정
    train_dataset, eval_dataset = data_processor.prepare_datasets()

    # 학습
    trainer = CustomTrainer(
        training_config=config["training"],
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trained_model = trainer.train()

    # 추론
    inferencer = Inferencer(inference_config=config["inference"], model=trained_model, tokenizer=tokenizer)
    inferencer.run_inference()


if __name__ == "__main__":
    main()

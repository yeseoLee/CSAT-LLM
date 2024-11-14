from data_loaders import DataLoader
from inference import InferenceModel
from loguru import logger
from model import ModelHandler
from trainer import CustomTrainer
import wandb

from .utils import (
    GoogleDriveManager,
    create_experiment_filename,
    load_config,
    log_config,
    set_logger,
    set_seed,
)


def main():
    # config, log, seed 설정
    config = load_config()
    set_logger(log_file=config["log"]["file"], log_level=config["log"]["level"])
    log_config(config)
    set_seed()

    # wandb 설정
    exp_name = create_experiment_filename(config)
    config["training"]["run_name"] = exp_name

    wandb.init(
        config=config,
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=exp_name,
    )

    try:
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
        inferencer = InferenceModel(
            data_config=config["data"],
            inference_config=config["inference"],
            model=trained_model,
            tokenizer=tokenizer,
        )
        output_filename = inferencer.run_inference()

    except Exception as e:
        logger.info(f"Error occurred: {e}")
        wandb.finish(exit_code=1)
    else:
        logger.info("Upload output to GDrive...")
        gdrive_manager = GoogleDriveManager()
        gdrive_manager.upload_output_csv(config["exp"]["username"], output_filename)
        wandb.finish()


if __name__ == "__main__":
    main()

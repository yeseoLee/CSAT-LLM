import argparse
from datetime import datetime
import os
import random

from loguru import logger
import numpy as np
import torch
import yaml
from zoneinfo import ZoneInfo


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(os.path.join("../config", args.config), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_logger(log_file="../log/file.log", log_level="DEBUG"):
    # 로거 설정
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        level=log_level,
        rotation="12:00",  # 매일 12시에 새로운 로그 파일 생성
        retention="7 days",  # 7일 후 로그 제거
    )


# config 확인
def log_config(config=load_config(), depth=0):
    if depth == 0:
        print("*" * 40)
    for k, v in config.items():
        prefix = ["\t" * depth, k, ":"]

        if isinstance(v, dict):
            print(*prefix)
            log_config(v, depth + 1)
        else:
            prefix.append(v)
            print(*prefix)
    if depth == 0:
        print("*" * 40)


def create_experiment_filename(config=load_config()):
    username = config["exp"]["username"]
    base_model = config["model"]["base_model"].replace("/", "_")
    train_path = config["data"]["train_path"]
    train_name = os.path.splitext(os.path.basename(train_path))[0]
    num_train_epochs = config["training"]["params"]["num_train_epochs"]
    learning_rate = config["training"]["params"]["learning_rate"]
    current_time = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%m%d%H%M")

    return f"{username}_{base_model}_{train_name}_{num_train_epochs}_{learning_rate}_{current_time}"

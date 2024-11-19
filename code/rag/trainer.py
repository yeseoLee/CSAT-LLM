from copy import deepcopy
import logging
import os
from typing import Tuple

from encoder import KobertBiEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import transformers
import wandb


# Initialize logger using logging module
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()  # get root logger


class Trainer:
    """Basic trainer"""

    def __init__(
        self,
        model,
        device,
        train_dataset,
        valid_dataset,
        num_epoch: int,
        batch_size: int,
        lr: float,
        betas: Tuple[float],
        num_warmup_steps: int,
        num_training_steps: int,
        valid_every: int,
        best_val_ckpt_path: str,  # Ensure this is passed as an argument
        collate_fn: callable = None,  # Make collate_fn optional
    ):
        # Initialize class variables
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps, num_training_steps
        )

        # Ensure that padding token id is accessed correctly
        padding_value = train_dataset.pad_token_id if hasattr(train_dataset, "pad_token_id") else 0

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            collate_fn=lambda x: custom_collator(x, padding_value=padding_value),
            num_workers=4,
            shuffle=True,
        )

        self.valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            collate_fn=lambda x: custom_collator(x, padding_value=padding_value),
            num_workers=4,
            shuffle=False,
        )

        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.valid_every = valid_every
        self.lr = lr
        self.betas = betas
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.best_val_ckpt_path = best_val_ckpt_path  # Ensure this is initialized
        self.best_val_optim_path = best_val_ckpt_path.split(".pt")[0] + "_optim.pt"

        self.start_ep = 1
        self.start_step = 1

    def ibn_loss(self, pred: torch.FloatTensor):
        """in-batch negative를 활용한 batch의 loss를 계산합니다."""
        bsz = pred.size(0)
        target = torch.arange(bsz).to(self.device)  # 주대각선이 answer
        return torch.nn.functional.cross_entropy(pred, target)

    def batch_acc(self, pred: torch.FloatTensor):
        """batch 내의 accuracy를 계산합니다."""
        bsz = pred.size(0)
        target = torch.arange(bsz)  # 주대각선이 answer
        return (pred.detach().cpu().max(1).indices == target).sum().float() / bsz

    def fit(self):
        """모델을 학습합니다."""
        wandb.init(
            project="kordpr",
            entity="lucas01",
            config={
                "batch_size": self.batch_size,
                "lr": self.lr,
                "betas": self.betas,
                "num_warmup_steps": self.num_warmup_steps,
                "num_training_steps": self.num_training_steps,
                "valid_every": self.valid_every,
            },
        )
        logger.debug("start training")
        self.model.train()  # 학습모드
        global_step_cnt = 0
        prev_best = None
        for ep in range(self.start_ep, self.num_epoch + 1):
            for step, batch in enumerate(tqdm(self.train_loader, desc=f"epoch {ep} batch"), 1):
                if ep == self.start_ep and step < self.start_step:
                    continue  # 중간부터 학습시키는 경우 해당 지점까지 복원

                self.model.train()  # 학습 모드
                global_step_cnt += 1
                p, p_mask = batch  # 질문 데이터 제거
                p, p_mask = p.to(self.device), p_mask.to(self.device)
                p_emb = self.model(p, p_mask)  # bsz x bert_dim
                pred = torch.matmul(p_emb, p_emb.T)  # bsz x bsz
                loss = self.ibn_loss(pred)
                acc = self.batch_acc(pred)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                log = {
                    "epoch": ep,
                    "step": step,
                    "global_step": global_step_cnt,
                    "train_step_loss": loss.cpu().item(),
                    "current_lr": float(self.scheduler.get_last_lr()[0]),  # parameter group 1개이므로
                    "step_acc": acc,
                }
                if global_step_cnt % self.valid_every == 0:
                    eval_dict = self.evaluate()
                    log.update(eval_dict)
                    if prev_best is None or eval_dict["valid_loss"] < prev_best:  # best val loss인 경우 저장
                        self.save_training_state(log)
                wandb.log(log)

    def evaluate(self):
        """모델을 평가합니다."""
        self.model.eval()  # 평가 모드
        loss_list = []
        sample_cnt = 0
        valid_acc = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                p, p_mask = batch
                p, p_mask = p.to(self.device), p_mask.to(self.device)
                p_emb = self.model(p, p_mask)  # bsz x bert_dim
                pred = torch.matmul(p_emb, p_emb.T)  # bsz x bsz
                loss = self.ibn_loss(pred)
                step_acc = self.batch_acc(pred)

                bsz = p.size(0)
                sample_cnt += bsz
                valid_acc += step_acc * bsz
                loss_list.append(loss.cpu().item() * bsz)
        return {
            "valid_loss": np.array(loss_list).sum() / float(sample_cnt),
            "valid_acc": valid_acc / float(sample_cnt),
        }

    def save_training_state(self, log_dict: dict) -> None:
        """모델, optimizer와 기타 정보를 저장합니다"""
        # Save the model checkpoint
        torch.save(self.model.state_dict(), self.best_val_ckpt_path)
        # Save optimizer state and other logs
        training_state = {
            "optimizer_state": deepcopy(self.optimizer.state_dict()),
            "scheduler_state": deepcopy(self.scheduler.state_dict()),
        }
        training_state.update(log_dict)
        torch.save(training_state, self.best_val_optim_path)
        logger.debug(f"saved optimizer/scheduler state into {self.best_val_optim_path}")

    def load_training_state(self) -> None:
        """모델, optimizer와 기타 정보를 로드합니다"""
        # Check if checkpoint exists
        if os.path.exists(self.best_val_ckpt_path):
            self.model.load_state_dict(torch.load(self.best_val_ckpt_path))
            logger.debug(f"Loaded model from {self.best_val_ckpt_path}")
        else:
            logger.debug("No checkpoint found, starting from scratch.")

        # Load optimizer and scheduler state
        if os.path.exists(self.best_val_optim_path):
            training_state = torch.load(self.best_val_optim_path)
            self.optimizer.load_state_dict(training_state["optimizer_state"])
            self.scheduler.load_state_dict(training_state["scheduler_state"])
            self.start_ep = training_state["epoch"]
            self.start_step = training_state["step"]
            logger.debug(f"Resumed training from epoch {self.start_ep} / step {self.start_step}")
        else:
            logger.debug("No optimizer state found, starting from scratch.")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    model = KobertBiEncoder()

    # 기존 데이터셋 로드 및 분할
    train_dataset = CustomDataset(csv_path="../../data/train.csv")
    train_data, valid_data = train_test_split(train_dataset.data, test_size=0.1)  # 10% 검증 데이터
    train_dataset = CustomDatasetFromData(train_data)
    valid_dataset = CustomDatasetFromData(valid_data)

    # Trainer 객체 초기화
    my_trainer = Trainer(
        model=model,
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        num_epoch=40,
        batch_size=96,  # 적절한 배치 사이즈
        lr=1e-5,
        betas=(0.9, 0.99),
        num_warmup_steps=1000,
        num_training_steps=100000,
        valid_every=30,
        best_val_ckpt_path="my_model.pt",
    )

    # 훈련 시작
    my_trainer.load_training_state()  # Check if checkpoint exists
    eval_dict = my_trainer.evaluate()  # Initial evaluation
    print(eval_dict)
    my_trainer.fit()  # Start training

import logging
import random
import traceback
from datetime import datetime

import numpy as np
import torch

from configs.arguments import TrainingArguments
from dataset.processor import MyDataset
from models.STD import STGDL
from trainer import Trainer
from utils.file_utils import ensure_dir


def main(config: TrainingArguments):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time = datetime.strftime(datetime.now(), "%m%d_%H%M%S")
    setup(config, time)

    try:
        logging.info("Loading dataset...")
        dataset = MyDataset(config)
        supports = [torch.Tensor(support).to(device) for support in dataset.supports]

        logging.info("Loading model...")
        model = STGDL(config, supports, dataset.scaler, device).to(device)
        print_parameter_count(model)
        trainer = Trainer(config, model, dataset, time)

        logging.info("Start Training.")
        # --load 0601_235959 加载已训练模型，可能是断点训练也可能是 debug/可视化
        if config.load != "":
            trainer.load(f"saved_dict/{config.load}.ckpt")

        if config.load == "":
            trainer.train(time)
            trainer.load(trainer.save_path)
        elif config.continue_training_epoch != -1:
            trainer.load(trainer.save_path)
            trainer.train(time, config.continue_training_epoch)

        trainer.test()
        print_parameter_count(model)
    except Exception as e:
        traceback.print_exc()


def setup(config, time):
    ensure_dir("saved_dict")
    ensure_dir(f"logs/{time}")

    # logging config
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s   %(levelname)s   %(message)s')
    logger = logging.getLogger()
    formats = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'logs/log{time}.txt')
    file_handler.setFormatter(formats)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    for x in config:
        logging.info(x)

    setup_seed(config.seed)


def print_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    total_req = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"All para: {total}, trainable: {total_req}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

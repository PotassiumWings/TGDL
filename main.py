import logging
import random
import traceback
from datetime import datetime

import numpy as np
import torch

from configs.arguments import TrainingArguments
from dataset.processor import MyDataset
from models.TGDL import TGDL
from trainer import Trainer
from utils.file_utils import ensure_dir


def main(config: TrainingArguments):
    torch.autograd.set_detect_anomaly(True)
    time = datetime.strftime(datetime.now(), "%m%d_%H%M%S")
    setup(config, time)

    try:
        logging.info("Loading dataset...")
        dataset = MyDataset(config)
        # 这里 supports 是接下来要被分解的 G，一般只有一个图，也即邻接矩阵
        # 按理说不应该放在 dataset 这里，但要算 DTW 距离，就放这了
        supports = dataset.supports
        scaler = dataset.scaler

        logging.info("Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TGDL(config, supports, scaler, device).to(device)
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

        if config.save_graph:
            trainer.visualize("subgraphs")

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

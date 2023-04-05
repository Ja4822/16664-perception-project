import cv2
import os.path as osp
import numpy as np
import pandas as pd
import pyrallis
from tqdm import trange

from dataclasses import asdict, dataclass
from utils import load_eval_data, set_seed, \
    save_config, load_config, gen_log_dir, ImageDataset

import torch
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


@dataclass
class EvalConfig:
    data_path: str = "data"
    model_path: str = "log"
    logdir: str = "log"
    device: str = "cuda:0"
    optimal: bool = True


@pyrallis.wrap()
def eval(cfg: EvalConfig):
    loaded_cfg = load_config(osp.join(cfg.model_path, "config.json"))
    loaded_cfg = Dict2Class(loaded_cfg)
    set_seed(loaded_cfg.seed)
    
    # save config
    LOG_DIR = gen_log_dir(loaded_cfg, "eval")
    if cfg.optimal:
        LOG_DIR = gen_log_dir(loaded_cfg, "eval_optimal")
    save_config(cfg, LOG_DIR)
    print(f"save evaluation results to {LOG_DIR}")
    
    # load eval data    
    eval_img_dirs = load_eval_data(cfg.data_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = osp.join(cfg.model_path, "model.pt")
    if cfg.optimal:
        model_path = osp.join(cfg.model_path, "model_optimal.pt")
    print(f"load trained model from {model_path}")

    model = torch.load(model_path).to(cfg.device)
    model.eval()
    
    eval_img_name = []
    preds = []

    for i in trange(len(eval_img_dirs), desc="Evaluating ..."):
        eval_img_dir = eval_img_dirs[i]
        eval_img = cv2.imread(eval_img_dir)
        eval_img = cv2.cvtColor(eval_img, cv2.COLOR_BGR2RGB)
        eval_img = transform(Image.fromarray(eval_img)).to(cfg.device)

        # img = eval_img.detach().cpu().numpy().transpose((1, 2, 0))
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # img = std * img + mean
        # img = np.clip(img, 0, 1)
        # plt.imshow(img, cmap="gray")
        # plt.savefig("eval.png")

        outputs = model(eval_img[None, ...])
        _, pred = torch.max(outputs, 1)
        pred = pred.detach().cpu().numpy()
        preds.append(pred[0])
        names = eval_img_dir.split("/")
        eval_img_name.append(osp.join(names[-2], names[-1][:-10]))

    classnum = 3
    for i in range(classnum):
        print(f"pred class id = {i}, num = {np.sum(np.array(preds) == i)}")

    eval_results = {"guid/image": eval_img_name, "label": preds}
    df = pd.DataFrame(eval_results)
    df.to_csv(osp.join(LOG_DIR, "pred_labels.csv"), index=False)


if __name__ == "__main__":
    eval()

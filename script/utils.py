import time
import json
import os
import os.path as osp
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import cv2
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, img_dirs, labels, transform=None, target_transform=None):
        self.img_dirs = img_dirs
        # self.labels = F.one_hot(torch.tensor(labels, dtype=int), 3)
        self.labels = torch.tensor(labels, dtype=int)
        # self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_dirs)
        
    def __getitem__(self, idx):
        # size = [b, w, h, 3]
        image = cv2.imread(self.img_dirs[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # label = self.labels[idx, :]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(Image.fromarray(image))
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def set_seed(seed: int, deterministic_torch: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(path: str, percent: float = 0.7):
    print(f"load data from {osp.join(path, 'labels.csv')}")
    classes = np.loadtxt(osp.join(path, 'labels.csv'), skiprows=1, dtype=str, delimiter=',')
    labels = classes[:, 1].astype(np.uint8)
    
    # shuffle data
    shuffle_idx = np.random.permutation(np.arange(labels.shape[0]))
    labels = labels[shuffle_idx]
    img_dirs = []
    for i in range(shuffle_idx.shape[0]):
        idx = shuffle_idx[i]
        img_dirs.append(osp.join(path, classes[idx, 0] + "_image.jpg"))
    
    # seperate data into train and test
    train_idx = int(len(img_dirs) * percent)
    train_img_dirs = img_dirs[:train_idx]
    train_labels   = labels[:train_idx]
    test_img_dirs = img_dirs[train_idx:]
    test_labels   = labels[train_idx:]
    print(f"data num = {len(img_dirs)}, train data num = {len(train_img_dirs)}, test data num = {len(test_img_dirs)}")
    return train_img_dirs, train_labels, test_img_dirs, test_labels


def load_eval_data(path: str):
    print(f"load data from {path}")
    logdirs = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                logdirs.append(osp.join(root, file))
    print(f"data num = {len(logdirs)}")
    return logdirs


def gen_log_dir(cfg, suffix: str = None):
    root = osp.dirname(osp.realpath(__file__))
    hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    logdir = osp.join(cfg.logdir, hms_time + "_s" + str(cfg.seed))
    if suffix is not None:
        logdir += "_" + suffix
    logdir = osp.join(root, logdir)
    return logdir


def save_config(cfg, path: str):
    json_cfg = json.dumps(cfg.__dict__)
    if not osp.exists(path):
        os.makedirs(path)
    with open(osp.join(path, "config.json"), "w") as f:
        f.write(json_cfg)


def load_config(path: str):
    with open(path) as f:
        config = json.load(f)
    return config

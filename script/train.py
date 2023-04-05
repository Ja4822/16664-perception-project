import os
import cv2
import os.path as osp
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pyrallis
from tqdm import trange

from dataclasses import asdict, dataclass
from utils import load_data, set_seed, \
    gen_log_dir, save_config, ImageDataset, load_eval_data

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainConfig:
    path: str = "data"
    device: str = "cuda:0"
    logdir: str = "log"
    suffix: str = None,
    seed: int = 0
    batch_size: int = 128
    lr: float = 1e-4
    epochs: int = 100
    train_percent: float = 0.7
    num_workers: int = 16


@pyrallis.wrap()
def train(cfg: TrainConfig):
    # set seed for reproducibility
    set_seed(cfg.seed)
    
    # save config
    LOG_DIR = gen_log_dir(cfg, cfg.suffix)
    save_config(cfg, LOG_DIR)
    print(f"save model to {LOG_DIR}")
    
    # load dataset
    train_img_dirs, train_labels, test_img_dirs, test_labels = load_data(cfg.path, cfg.train_percent)
    classnum = 3
    for i in range(classnum):
        print(f"pred class id = {i}, num = {np.sum(np.array(train_labels) == i)}")
    for i in range(classnum):
        print(f"pred class id = {i}, num = {np.sum(np.array(test_labels) == i)}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform = None
    train_data = ImageDataset(train_img_dirs, train_labels, transform, target_transform)
    train_dataloader = DataLoader(train_data, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    
    test_data = ImageDataset(test_img_dirs, test_labels, transform, target_transform)
    test_dataloader = DataLoader(test_data, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # ResNet50 model, fix all the parameters except the last layer for transfer learning
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT').to(cfg.device)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 3)).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss = nn.CrossEntropyLoss()

    # logger
    writer = SummaryWriter(osp.join(LOG_DIR, "tb"))
    total_acc_max = 0
    best_model_epoch = 0

    for n in trange(cfg.epochs, desc="Training ..."):
        # train
        train_loss = []
        train_acc = []
        model.train()
        for train_imgs, train_labels in train_dataloader:
            train_imgs = train_imgs.to(cfg.device)
            train_labels = train_labels.to(cfg.device)

            optimizer.zero_grad()
            outputs = model(train_imgs)
            _, preds = torch.max(outputs, 1)
            batch_loss = loss(outputs, train_labels)
            batch_loss.backward()
            optimizer.step()

            train_loss.append(batch_loss.item() * train_imgs.size(0))
            batch_acc = torch.sum(preds == train_labels.data) / train_imgs.size(0)
            train_acc.append(batch_acc.detach().cpu().numpy())

        # logger
        writer.add_scalar('train/epoch_loss', np.mean(train_loss), global_step=n)
        writer.add_scalar('train/epoch_acc', np.mean(train_acc), global_step=n)
        
        # test
        test_loss = []
        test_acc = []
        model.eval()
        for test_imgs, test_labels in test_dataloader:
            test_imgs = test_imgs.to(cfg.device)
            test_labels = test_labels.to(cfg.device)
            outputs = model(test_imgs)
            _, preds = torch.max(outputs, 1)
            test_loss.append(loss(outputs, test_labels).item() * test_imgs.size(0))
            batch_acc = torch.sum(preds == test_labels.data) / test_imgs.size(0)
            test_acc.append(batch_acc.detach().cpu().numpy())
        
        # logger
        writer.add_scalar('test/epoch_loss', np.mean(test_loss), global_step=n)
        writer.add_scalar('test/epoch_acc', np.mean(test_acc), global_step=n)
    
        # save optimal model
        total_acc = (1 - cfg.train_percent) * np.mean(train_acc) + \
                    cfg.train_percent * np.mean(test_acc)
        if total_acc > total_acc_max:
            torch.save(model, osp.join(LOG_DIR, "model_optimal.pt"))
            total_acc_max = total_acc
            best_model_epoch = n
        writer.add_scalar('test/total_acc', total_acc, global_step=n)
        writer.add_scalar('test/best_model_epoch', best_model_epoch, global_step=n)
        
        # save model at each epoch
        torch.save(model, osp.join(LOG_DIR, "model.pt"))
    
    # train_imgs, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_imgs.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_imgs[0].squeeze()
    # label = train_labels[0]
    # img = img.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # img = std * img + mean
    # img = np.clip(img, 0, 1)
    # plt.imshow(img, cmap="gray")
    # plt.savefig("text.png")

if __name__ == "__main__":
    train()

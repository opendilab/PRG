import os
from easydict import EasyDict
import torch
from torch.utils.data import Dataset
import ipdb
import wandb
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import random
from accelerate import Accelerator
from data import build_loader
from models import build_model
from torch_tool import build_optimizer, build_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import os
import matplotlib
import cv2
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import numpy as np
import torchvision
from grl.utils.model_utils import save_model, load_model
from rich.progress import track
import torchvision

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

@torch.no_grad()
def validate(accelerator,model, val_loader):
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    model.train()
    for images, targets in track(val_loader, disable=not accelerator.is_local_main_process):
        outputs = model(images,False)
        outputs, targets = accelerator.gather_for_metrics((outputs, targets))
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(acc1[0], outputs.size(0))
        top5.update(acc5[0], outputs.size(0))
    return top1.avg

def train(config, accelerator):
    if accelerator.is_main_process:
        wandb_mode = "online" if accelerator.state.num_processes > 1 else "offline"
        wandb.init(
            project=config.PROJECT_NAME,
            config=config,
            mode=wandb_mode  
        )
        if hasattr(config, "extra"):
            wandb.run.name=config.extra
            wandb.run.save()
    accelerator.wait_for_everyone()
    data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    model = build_model(config)
    if (
            hasattr(config.DATA, "checkpoint_path")
            and config.DATA.checkpoint_path is not None
        ):
        load_model(
                path=config.DATA.checkpoint_path,
                model=model,
                optimizer=None,
                prefix="GenerativeClassify",
        )
    (
        model.grlEncoder.diffusionModel.model,
        model.grlHead,
        data_loader_val,
    ) = accelerator.prepare(
        model.grlEncoder.diffusionModel.model,
        model.grlHead,
        data_loader_val,
    )
    acc1=validate(accelerator, model, data_loader_val)
    accelerator.print("acc",acc1)
               
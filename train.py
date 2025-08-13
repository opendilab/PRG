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



def train_epoch(accelerator, model, criterion, data_loader, optimizer,lr_scheduler, epoch):
    model.train()
    num_steps=len(data_loader)
    idx=0
    for samples, targets in track(
        data_loader,
        description=f"Processing Train epoch {epoch}",
        disable=not accelerator.is_main_process,
    ):
        outputs = model(samples)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)
        idx+=1
        # max_param_val, max_grad_val, min_grad_val = find_max_param_and_grad(model)
        if accelerator.is_main_process:
            wandb.log(
                {"train/loss": loss, "train/epoch": epoch},
                commit=True,
            )
    return 0

def train_flow_matching(accelerator, model, data_loader, optimizer, epoch):
    model.train()
    for samples, targets in track(
        data_loader
    ):
        loss = model.matchingLoss(samples)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        wandb.log(
            {"train/loss": loss, "train/epoch": epoch},
            commit=True,
        )
    return 0

def train_icfm_flow_matching(accelerator, model, data_loader, optimizer, epoch):
    model.train()
    for samples, targets in track(
        data_loader
    ):
        x0 = model.grlEncoder.diffusionModel.gaussian_generator(samples.shape[0]).to(samples.device)
        loss = model.matchingLoss(x0=x0,x1=samples)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        
        wandb.log(
            {"train/loss": loss, "train/epoch": epoch},
            commit=True,
        )
    return 0


@torch.no_grad()
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

@torch.no_grad()
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
def validate(accelerator, model, val_loader, criterion, epoch,mixup_fn=None):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    model.eval()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            if mixup_fn is not None:
                images, targets_mixup=mixup_fn(images, targets)
            else :
                targets_mixup=targets
            outputs = model(images,False)
            loss = criterion(outputs, targets_mixup)
            outputs, targets = accelerator.gather_for_metrics((outputs, targets))
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), outputs.size(0))
            top1.update(acc1[0], outputs.size(0))
            top5.update(acc5[0], outputs.size(0))
       
        if accelerator.is_main_process:
            wandb.log(
                {
                    f"eval/acc1": top1.avg,
                    f"eval/acc5": top5.avg,
                    f"eval/loss": losses.avg,
                    f"eval/epoch": epoch,
                },
                commit=False,
            )
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
    
    if config.TRAIN.method == "Pretrain":
        accelerator.print("Pretrain")
        ### Load the data
        data_loader_train, data_loader_val, mixup_fn = build_loader(config)
        model = build_model(config)
        optimizer = build_optimizer(config, model)
        
        ### Load the model
        if (
                hasattr(config.DATA, "checkpoint_path")
                and config.DATA.checkpoint_path is not None
            ):
            diffusion_model_train_epoch = load_model(
                    path=config.DATA.checkpoint_path,
                    model=model.grlEncoder.diffusionModel.model,
                    optimizer=None,
                    prefix="DiffusionModel_Pretrain",
                )
        else:
                diffusion_model_train_epoch=0
                
        ### Prepare the model
        (
            model.grlEncoder.diffusionModel.model,
            data_loader_train,
            data_loader_val,
            optimizer,
        ) = accelerator.prepare(
            model.grlEncoder.diffusionModel.model,
            data_loader_train,
            data_loader_val,
            optimizer,
        )
        
        ### Train the model
        for epoch in range(config.TRAIN.iteration):
            if diffusion_model_train_epoch >= epoch:
                continue
            
                
            if (epoch + 1) % config.TEST.checkpoint_freq == 0:
                if accelerator.is_local_main_process:
                    save_model(
                        config.DATA.checkpoint_path,
                        model.grlEncoder.diffusionModel.model,
                        optimizer,
                        epoch,
                        "DiffusionModel_Pretrain",
                    )
                accelerator.wait_for_everyone()
                    
            if config.MODEL.model_type in ["ICFM","OT"]:
                train_icfm_flow_matching(accelerator, model, data_loader_train, optimizer, epoch)
            elif config.MDOEL.model_type=="Diff":
                train_flow_matching(accelerator, model, data_loader_train, optimizer, epoch)
            else:
                raise NotImplementedError("Model type not implemented")
            
        
    if config.TRAIN.method == "Finetune" or config.TRAIN.method == "Pretrain":
        accelerator.print("Finetune")
        data_loader_train, data_loader_val, mixup_fn = build_loader(config)
        model = build_model(config)
        optimizer = build_optimizer(config, model)
        
        if (
                hasattr(config.DATA, "checkpoint_path")
                and config.DATA.checkpoint_path is not None
            ):
                load_model(
                        path=config.DATA.checkpoint_path,
                        model=model.grlEncoder.diffusionModel.model,
                        optimizer=None,
                        prefix="DiffusionModel_Pretrain",
                )
                
        if config.TRAIN.loss_function == "CrossEntropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif config.TRAIN.loss_function == "LabelSmoothingCrossEntropy":
            criterion = LabelSmoothingCrossEntropy(smoothing=config.TRAIN.label_smoothing)
        elif config.TRAIN.loss_function == "SoftTargetCrossEntropy":
            criterion = SoftTargetCrossEntropy()
        else:
            raise NotImplementedError
        (
            model.grlEncoder.diffusionModel.model,
            model.grlHead,
            data_loader_train,
            data_loader_val,
            optimizer,
        ) = accelerator.prepare(
            model.grlEncoder.diffusionModel.model,
            model.grlHead,
            data_loader_train,
            data_loader_val,
            optimizer,
        )
        lr_scheduler= build_scheduler(config, optimizer,len(data_loader_train))
        for epoch in range(config.TRAIN.iteration):
            if (epoch+1) % config.TEST.eval_freq == 0:
                validate(accelerator, model, data_loader_val, criterion, epoch,mixup_fn)
            train_epoch(accelerator, model, criterion, data_loader_train, optimizer, lr_scheduler,epoch)
        
            if (epoch + 1) % config.TEST.checkpoint_freq == 0:
                save_model(
                    config.DATA.checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    "GenerativeClassify",
                )
                accelerator.wait_for_everyone()
                    
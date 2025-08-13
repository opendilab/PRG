import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .TinyImageNet import TinyImageNet

from torchvision.transforms import InterpolationMode


def _pil_interp(method):
    if method == "bicubic":
        return InterpolationMode.BICUBIC
    elif method == "lanczos":
        return InterpolationMode.LANCZOS
    elif method == "hamming":
        return InterpolationMode.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return InterpolationMode.BILINEAR


import timm.data.transforms as timm_transforms

timm_transforms._pil_interp = _pil_interp

def build_loader(config,if_analyse=False):
    dataset_train = build_dataset(is_train=True, config=config,if_analyse=if_analyse)
    dataset_val = build_dataset(is_train=False, config=config,if_analyse=if_analyse)


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.DATA.batch_size,
        shuffle=False if if_analyse else True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.batch_size,
        shuffle=False ,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    # if hasattr(config.DATA, "MIXUP"):
    #     mixup_fn = Mixup(
    #         mixup_alpha=config.DATA.MIXUP.mixup,
    #         cutmix_alpha=config.DATA.MIXUP.cutmix,
    #         cutmix_minmax=config.DATA.MIXUP.cutminmax,
    #         prob=config.DATA.MIXUP.mixup_prob,
    #         switch_prob=config.DATA.MIXUP.mixup_switch_prob,
    #         mode=config.DATA.MIXUP.mixup_mode,
    #         label_smoothing=config.TRAIN.label_smoothing,
    #         num_classes=config.MODEL.classes,
    #     )
    # else:
    #     mixup_fn = None
    return data_loader_train, data_loader_val, None


def build_dataset(is_train, config,if_analyse):
    if if_analyse :
        transform = transforms.Compose(
            [
            transforms.Resize(config.DATA.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        transform = build_transform(is_train, config)
    if config.DATA.dataset == "Tinyimagenet":
        if is_train:
            dataset = TinyImageNet(config.DATA.dataset_path, True, transform)
        else:
            dataset = TinyImageNet(config.DATA.dataset_path, False, transform)
    elif config.DATA.dataset == "Imagenet":
        if is_train:
            dataset = datasets.ImageFolder(
                os.path.join(config.DATA.dataset_path, "train"),
                transform=transform,
            )
        else:
            dataset = datasets.ImageFolder(
                os.path.join(config.DATA.dataset_path, "val"), transform=transform
            )
    elif config.DATA.dataset == "CIFAR-10":
        if is_train:
            dataset = datasets.CIFAR10(
                root=config.DATA.dataset_path,
                train=True,
                download=True,
                transform=transform,
            )
        else:
            dataset = datasets.CIFAR10(
                root=config.DATA.dataset_path,
                train=False,
                download=True,
                transform=transform,
            )  
    else:
        raise NotImplementedError("We only support ImageNet Now.")
    return dataset


def build_transform(is_train, config):
    if config.TRAIN.method in ["Finetune"]:
        resize_im = config.DATA.img_size > 32
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=config.DATA.img_size,
                is_training=True,
                color_jitter=config.DATA.AUG.color_jitter if config.DATA.AUG.color_jitter > 0 else None,
                auto_augment=config.DATA.AUG.auto_augment if config.DATA.AUG.auto_augment != 'none' else None,
                re_prob=config.DATA.AUG.reprob,
                re_mode=config.DATA.AUG.remode,
                re_count=config.DATA.AUG.recount,
                interpolation=config.DATA.AUG.interpolation,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(config.DATA.img_size, padding=4)
            return transform

        t = []
        if resize_im:
            if config.TEST.crop:
                size = int((256 / 224) * config.DATA.img_size)
                t.append(
                    transforms.Resize(size, interpolation=_pil_interp(config.DATA.AUG.interpolation)),
                    # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(config.DATA.img_size))
            else:
                t.append(
                    transforms.Resize((config.DATA.img_size, config.DATA.img_size),
                                    interpolation=_pil_interp(config.DATA.AUG.interpolation))
                )

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)
    
    elif config.TRAIN.method in ["Pretrain"]:
        resize_im = config.DATA.img_size > 32
        transform = create_transform(
                input_size=config.DATA.img_size,
                is_training=True,
                color_jitter=config.DATA.AUG.color_jitter if config.DATA.AUG.color_jitter > 0 else None,
                auto_augment=config.DATA.AUG.auto_augment if config.DATA.AUG.auto_augment != 'none' else None,
                re_prob=config.DATA.AUG.reprob,
                re_mode=config.DATA.AUG.remode,
                re_count=config.DATA.AUG.recount,
                interpolation=config.DATA.AUG.interpolation,
            )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.img_size, padding=4)
        return transform

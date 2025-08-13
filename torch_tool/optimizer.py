import torch


def build_optimizer(config, model):
    if config.TRAIN.optimizer_type == "adam":
        if config.TRAIN.method == "Finetune":
            optimizer = torch.optim.Adam(
                [
                    {"params": model.grlHead.parameters()},
                    {"params": model.grlEncoder.diffusionModel.model.parameters()},
                ],
                lr=config.TRAIN.lr,
            )
        elif config.TRAIN.method == "Pretrain":
            optimizer = torch.optim.Adam(
                model.grlEncoder.diffusionModel.model.parameters(),
                lr=config.TRAIN.lr,
            )    
    if config.TRAIN.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            [
                    {"params": model.grlHead.parameters()},
                    {"params": model.grlEncoder.diffusionModel.model.parameters()},
            ],
            eps=config.TRAIN.OPTIMIZER.eps, betas=config.TRAIN.OPTIMIZER.betas,
            lr=config.TRAIN.lr, weight_decay=config.TRAIN.OPTIMIZER.weight_decay)
    return optimizer






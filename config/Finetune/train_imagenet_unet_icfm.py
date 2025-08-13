import wandb
from easydict import EasyDict
from accelerate import Accelerator
from GenerativeClassify.classify.improved_classify.train import train

def make_config(device):
    model_type="ICFM"
    method="Finetune"
    type=f"GenerativeClassifyUNet_{model_type}"
    classes = 1000
    image_size = 64
    project_name = f"z_{model_type}_{method}_Imagnet"
    config = EasyDict(
        dict(
            PROJECT_NAME=project_name,
            extra="icfm-10-cut",
            DEVICE=device,
            DATA=dict(
                batch_size=128,
                classes=classes,
                img_size=image_size,
                dataset_path="/mnt/afs/zhangjinouwen/Dataset/imagenet",
                checkpoint_path=f"/root/Model/ImageNet",
                video_save_path=f"/root/Model/ImageNet",
                dataset="Imagenet",
                AUG=dict(
                    interpolation="bicubic",
                    color_jitter=0.4,
                    auto_augment="rand-m9-mstd0.5-inc1",
                    reprob=0.25,
                    remode="pixel",
                    recount=1,
                ),
            ),
            MODEL=dict(
                method=method,
                type=type,
                t_span=32,
                t_cutoff=20,
                image_size=image_size,
                classes=classes,
                model_type=model_type,
                diffusion_model=dict(
                    device=device,
                    x_size=(3, image_size, image_size),
                    alpha=1.0,
                    solver=dict(
                        type="ODESolver",
                        args=dict(
                            library="torchdiffeq_adjoint",
                        ),
                    ),
                    path=dict(
                        sigma=0.0,
                    ),
                    model=dict(
                        type="velocity_function",
                        args=dict(
                            backbone=dict(
                                type=type,
                                args={},
                            ),
                        ),
                    ),
                ),
            ),
            TRAIN=dict(
                method=method,
                loss_function="LabelSmoothingCrossEntropy", #LabelSmoothingCrossEntropy or SoftTargetCrossEntropy
                label_smoothing=0.1,
                training_loss_type="flow_matching",
                
                optimizer_type="adamw",
                lr=1.25e-4,
                warmup_lr=1.25e-07,
                min_lr=1.25e-6,
                
                
                iteration=200,
                warmup_iteration=5,
                decay_iteration=5,
                device=device,
                OPTIMIZER=dict(
                    eps=1e-08,
                    betas=(0.9, 0.999),
                    momentum=0.9,
                    weight_decay=0.05,
                ),
                LR_SCHEDULER=dict(
                    name="cosine",
                    decay_rate=0.1,
                    warmup_prefix=True,
                    gamma=0.1,
                    multisteps=[],
                ),
            ),
            TEST=dict(
                seed=0,
                crop=True,
                eval_freq=10,
                generative_freq=100,
                checkpoint_freq=5,
            ),
        )
    )
    return config


if __name__ == "__main__":
    accelerator = Accelerator()
    config = make_config(accelerator.device)
    num_processes = accelerator.num_processes
    config.TRAIN.lr=config.TRAIN.lr*num_processes*config.DATA.batch_size/512
    config.TRAIN.warmup_lr=config.TRAIN.warmup_lr*num_processes*config.DATA.batch_size/512
    config.TRAIN.min_lr=config.TRAIN.min_lr*num_processes*config.DATA.batch_size/512
    train(config, accelerator)

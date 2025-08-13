import torch
from grl.neural_network.unet import unet_2D
from grl.neural_network import register_module
from grl.generative_models.conditional_flow_model.independent_conditional_flow_model import (
    IndependentConditionalFlowModel,
)
import torch.nn as nn


class Unet_64(nn.Module):
    def __init__(self):
        super(Unet_64, self).__init__()
        self.unet = unet_2D(
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=tuple([4, 8]),
            dropout=0,
            channel_mult=(1, 2, 3, 4),
            num_classes=None,
            use_checkpoint=False,
            num_heads=4,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.unet(x, t, condition)
        return x
 

class Unet_32(nn.Module):
    def __init__(self):
        super(Unet_32, self).__init__()
        self.unet = unet_2D(
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=tuple([2, 4]),
            dropout=0,
            channel_mult=(1, 2, 2, 2),
            num_classes=None,
            use_checkpoint=False,
            num_heads=4,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.unet(x, t, condition)
        return x


class classifyHead(nn.Module):
    def __init__(self, config):
        super(classifyHead, self).__init__()
        self.mypool=nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)) ,
            nn.Flatten(),
        )
        self.classifier=nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Linear(768,  config.classes, bias=False),
        )
        self.config = config

    def forward(self, x):
        x = self.mypool(x)
        x = self.classifier(x)
        return x

class classifyHeadImagenet(nn.Module):
    def __init__(self, config):
        super(classifyHead, self).__init__()
        self.mypool=nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 32)) ,
            nn.Flatten(),
        )
        self.classifier=nn.Sequential(
            nn.LayerNorm(3072),
            nn.Linear(3072, 2048),
            nn.Tanh(),
            nn.Linear(2048,  config.classes, bias=False),
        )
        self.config = config

    def forward(self, x):
        x = self.mypool(x)
        x = self.classifier(x)
        return x

class generativeEncoder(nn.Module):
    def __init__(self, config):
        super(generativeEncoder, self).__init__()
        if config.image_size == 64:
            register_module(Unet_64, "GenerativeClassifyUNet_ICFM")
        elif config.image_size == 32:
            register_module(Unet_32, "GenerativeClassifyUNet_ICFM")
        self.diffusionModel = IndependentConditionalFlowModel(config.diffusion_model)
        self.config = config

    def sample_forward_process(self):
        self.diffusionModel.model.eval()
        import torchvision
        t_span = torch.linspace(0.0, 1.0, 32).to(self.config.diffusion_model.device)
        x_t = self.diffusionModel.sample_forward_process(t_span=t_span, batch_size=4)[
            -1, ...
        ]
        return x_t

    def sample_backward_process(self, x, with_grad=False):
        if hasattr(self.config, "t_cutoff"):
            t_span = torch.linspace(0.0, 1.0, self.config.t_span, device=x.device)[self.config.t_cutoff:]
        else :
            t_span = torch.linspace(0.0, 1.0, self.config.t_span, device=x.device)
        x_t = self.diffusionModel.forward_sample(
            x=x, t_span=t_span, with_grad=with_grad
        )
        return x_t


class generativeClassify(nn.Module):
    def __init__(self, config):
        super(generativeClassify, self).__init__()
        self.grlEncoder = generativeEncoder(config)
        self.grlHead = classifyHead(config)
        self.config = config

    def forward(self, x,with_grad=True):
        images = self.grlEncoder.sample_backward_process(x=x, with_grad=with_grad)
        output = self.grlHead(images)
        return output

    def matchingLoss(self,x0,x1):
        return self.grlEncoder.diffusionModel.flow_matching_loss(x0=x0,x1=x1)

    def samplePicture(self):
        return self.grlEncoder.sample_forward_process()

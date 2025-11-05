import torch.nn as nn


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.type
    if model_type == "GenerativeClassifyUNet_Diff":
        from .generativeClassify.unet_diff import generativeClassify

        model = generativeClassify(config.MODEL)
    elif model_type == "GenerativeClassifyUNet_ICFM":
        from .generativeClassify.unet_icfm import generativeClassify
        
        model = generativeClassify(config.MODEL)
    elif model_type == "GenerativeClassifyUNet_OT":
        from .generativeClassify.unet_ot import generativeClassify

        model = generativeClassify(config.MODEL)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    return model

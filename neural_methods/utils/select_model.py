import neural_methods.model as model

def get_model(config):
    if config.MODEL.NAME == 'mae_vit_base_patch16_dec512d8b':
        _model = model.MAEVit.mae_vit_base_patch16_dec512d8b()
    elif config.MODEL.NAME == 'ft_vit_base_patch16':
        _model = model.Vit.ft_vit_base_patch16()
    elif config.MODEL.NAME == 'convnextv2_femto':
        _model = model.ConvNeXtV2.convnextv2_femto()
    elif config.MODEL.NAME == 'resnet18':
        _model = model.ResNet.resnet18()
    else:
        raise ValueError(f"Unknown model name {config.MODEL.NAME}")
    return _model
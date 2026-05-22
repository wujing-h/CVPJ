import torch


def _make_optimizer(cfg_stage, params):
    if cfg_stage.OPTIMIZER_NAME == 'SGD':
        return getattr(torch.optim, cfg_stage.OPTIMIZER_NAME)(params, momentum=cfg_stage.MOMENTUM)
    if cfg_stage.OPTIMIZER_NAME == 'AdamW':
        return torch.optim.AdamW(params, lr=cfg_stage.BASE_LR, weight_decay=cfg_stage.WEIGHT_DECAY)
    return getattr(torch.optim, cfg_stage.OPTIMIZER_NAME)(params)


def _add_trainable_param(params, cfg_stage, key, value):
    lr = cfg_stage.BASE_LR
    weight_decay = cfg_stage.WEIGHT_DECAY
    if "bias" in key:
        lr = cfg_stage.BASE_LR * cfg_stage.BIAS_LR_FACTOR
        weight_decay = cfg_stage.WEIGHT_DECAY_BIAS
    if cfg_stage.LARGE_FC_LR:
        if "classifier" in key or "arcface" in key:
            lr = cfg_stage.BASE_LR * 2
            print('Using two times learning rate for fc ')
    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


def make_optimizer_1stage(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if "prompt_learner" in key:
            lr = cfg.SOLVER.STAGE1.BASE_LR
            weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    return _make_optimizer(cfg.SOLVER.STAGE1, params)


def make_optimizer_2stage_joint(cfg, model, center_criterion):
    prompt_params = []
    image_params = []
    prompt_lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.TEXT_LR_FACTOR
    for key, value in model.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(True)
            lr = prompt_lr
            weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
            prompt_params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            continue
        if not value.requires_grad:
            continue
        _add_trainable_param(image_params, cfg.SOLVER.STAGE2, key, value)

    optimizer_prompt = _make_optimizer(cfg.SOLVER.STAGE1, prompt_params)
    optimizer_image = _make_optimizer(cfg.SOLVER.STAGE2, image_params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)
    return optimizer_prompt, optimizer_image, optimizer_center


def make_optimizer_3stage(cfg, model, center_criterion):
    image_params = []
    for key, value in model.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        _add_trainable_param(image_params, cfg.SOLVER.STAGE3, key, value)

    optimizer_image = _make_optimizer(cfg.SOLVER.STAGE3, image_params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE3.CENTER_LR)
    return optimizer_image, optimizer_center

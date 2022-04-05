import torch
from torch.utils import data

import numpy as np


def construct_group_optimizer(model, args):
    """ model.encoder as the base parameters
        could be:
        1. pretrained word embeddings
        2. pretrained bert models
    """
    lr_x = args.lr * args.lr_mu

    encoder_param_ids = list(map(id, model.encoder.parameters()))

    other_params = filter(
        lambda p: id(p) not in encoder_param_ids,
        model.parameters()
    )

    param_groups = [
        {"params": other_params, "lr": args.lr},
        {"params": model.encoder.parameters(), "lr": lr_x},
    ]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("No such optimizer: {}".format(args.optimizer))
    return optimizer


def construct_optimizer(model, args):
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("No such optimizer: {}".format(args.optimizer))
    return optimizer


def construct_lr_scheduler(optimizer, args):
    if args.scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.scheduler == "CosLR":
        CosLR = torch.optim.lr_scheduler.CosineAnnealingLR
        lr_scheduler = CosLR(
            optimizer, T_max=args.epoches, eta_min=1e-8
        )
    elif args.scheduler == "CosLRWR":
        CosLRWR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        lr_scheduler = CosLRWR(
            optimizer, T_0=args.step_size
        )
    elif args.scheduler == "CyclicLR":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=0.0,
            max_lr=args.lr,
            step_size_up=args.step_size,
        )
    elif args.scheduler == "WSQuadLR":
        # LambdaLR: quadratic
        def lr_warm_start_quad(t, T0=args.ws_step, T_max=args.epoches):
            # T0 = int(0.1 * T_max)
            if t <= T0:
                return 1.0 * (t + 1e-6) / T0
            else:
                return (1.0 - 1.0 * (t - T0) / (T_max - T0)) ** 2

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_warm_start_quad,
        )
    elif args.scheduler == "WSStepLR":
        # LambdaLR: step lr
        def lr_warm_start_step(
            t, T0=args.ws_step,
            step_size=args.step_size, gamma=args.gamma
        ):
            if t <= T0:
                return 1.0 * (t + 1e-6) / T0
            else:
                return gamma ** int((t - T0) / step_size)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_warm_start_step,
        )
    elif args.scheduler == "WSCosLR":
        # LambdaLR: coslr
        def lr_warm_start_cos(
            t, T0=args.ws_step, T_max=args.epoches
        ):
            if t <= T0:
                return 1.0 * (t + 1e-6) / T0
            else:
                return (np.cos((t - T0) / (T_max - T0) * np.pi) + 1.0) / 2.0
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_warm_start_cos,
        )
    else:
        raise ValueError("No such scheduler: {}".format(args.scheduler))
    return lr_scheduler

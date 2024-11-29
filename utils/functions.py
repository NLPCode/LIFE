# -*- coding: utf-8 -*-
# @Author  : He Xingwei
# @Time : 2022/3/7 11:46
import torch
import numpy as np
import random
from torch.optim import AdamW, Adam
from utils.lamb import Lamb

def set_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
        
def sum_main(x, opt):
    if opt.world_size > 1:
        # torch.distributed.reduce(x, 0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x

def get_optimizer(optimizer, model: torch.nn.Module, weight_decay: float = 0.0, lr: float = 0, adam_epsilon=1e-8) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if optimizer == "adamW":
        return AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    elif optimizer == "lamb":
        return Lamb(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    else:
        raise Exception("optimizer {0} not recognized! Can only be lamb or adamW".format(optimizer))




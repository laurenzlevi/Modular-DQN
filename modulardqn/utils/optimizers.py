import torch.optim.adadelta

optimizers = {
    'Adadelta': torch.optim.Adadelta,
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'SparseAdam': torch.optim.SparseAdam,
    'Adamax': torch.optim.Adamax,
    'ASGD': torch.optim.ASGD,
    'LBFGS': torch.optim.LBFGS,
    'NAdam': torch.optim.NAdam,
    'RAdam': torch.optim.RAdam,
    'RMSProp': torch.optim.RMSprop,
    'Rprop': torch.optim.Rprop,
    'SGD': torch.optim.SGD
}


def make_optimizer(optimizer: str, parameters, lr: float) -> torch.optim.Optimizer:
    if optimizer not in optimizers.keys():
        raise ValueError(f"Unknown optimizer {optimizer}")

    return optimizers[optimizer](parameters, lr=lr)

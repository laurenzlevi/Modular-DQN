import torch

loss_functions = {
    'L1': torch.nn.functional.l1_loss,
    'MSE': torch.nn.functional.mse_loss,
    'CrossEntropy': torch.nn.functional.cross_entropy,
    'CTC': torch.nn.functional.ctc_loss,
    'NLL': torch.nn.functional.nll_loss,
    'PoissonNLL': torch.nn.functional.poisson_nll_loss,
    'GaussianNLL': torch.nn.functional.gaussian_nll_loss,
    'KLDiv': torch.nn.functional.kl_div,
    'BCE': torch.nn.functional.binary_cross_entropy,
    'Huber': torch.nn.functional.huber_loss,
    'SmoothL1': torch.nn.functional.smooth_l1_loss,
    'SoftMargin': torch.nn.functional.soft_margin_loss
}


def make_loss_func(loss_func: str):
    if loss_func not in loss_functions.keys():
        raise ValueError(f"Unknown loss function {loss_func}")

    return loss_functions[loss_func]

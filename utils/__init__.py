from torch.optim import SGD, Adam, RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR


def get_optimizer(hparams, models):
    eps = 1e-8
    parameters = []
    for model in models:
        parameters += list(model.parameters())
    if hparams['optimizer'] == 'sgd':
        optimizer = SGD(parameters, lr=hparams['lr'],
                        momentum=hparams['momentum'], weight_decay=hparams['weight_decay'])
    elif hparams['optimizer'] == 'adam':
        optimizer = Adam(parameters, lr=hparams['lr'], eps=eps,
                         weight_decay=hparams['weight_decay'])
    elif hparams['optimizer'] == 'radam':
        optimizer = RAdam(parameters, lr=hparams['lr'], eps=eps,
                          weight_decay=hparams['weight_decay'])
    # elif hparams.optimizer == 'ranger':
    #     optimizer = Ranger(parameters, lr=hparams.lr, eps=eps,
    #                       weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer


def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams['lr_scheduler'] == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams['decay_step'],
                                gamma=hparams['decay_gamma'])
    elif hparams['lr_scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams['num_epochs'], eta_min=eps)
    elif hparams['lr_scheduler'] == 'poly':
        scheduler = LambdaLR(optimizer,
                             lambda epoch: (1-epoch/hparams['num_epochs'])**hparams['poly_exp'])
    else:
        raise ValueError('scheduler not recognized!')

    # if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
    #     scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier,
    #                                        total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)

    return scheduler


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
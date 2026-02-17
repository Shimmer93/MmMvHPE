import sys
import importlib

from misc.lr_scheduler import LinearWarmupCosineAnnealingLR

def import_with_str(module, name):
    if module not in sys.modules:
        importlib.import_module(module)
    return getattr(sys.modules[module], name)

def create_model(model_name, model_params, eval=False):
    if model_params is None:
        model_params = {}
    model_class = import_with_str('models', model_name)
    model = model_class(**model_params)
    if eval:
        model.eval()
        model.requires_grad_(False)
    return model

def create_loss(loss_name, loss_params):
    if loss_params is None:
        loss_params = {}
    try:
        loss_class = import_with_str('losses', loss_name)
        loss = loss_class(**loss_params)
    except:
        loss_class = import_with_str('torch.nn', loss_name)
        loss = loss_class(**loss_params)
    return loss

def create_metric(metric_name, metric_params):
    if metric_params is None:
        metric_params = {}
    metric_class = import_with_str('metrics', metric_name)
    metric = metric_class(**metric_params)
    return metric

def create_optimizer(optim_name, optim_params, mparams):
    if optim_params is None:
        optim_params = {}
    optim_class = import_with_str('torch.optim', optim_name)
    optimizer = optim_class(mparams, **optim_params)
    return optimizer
    
def create_scheduler(sched_name, sched_params, optimizer):
    if sched_params is None:
        sched_params = {}
    if sched_name == 'LinearWarmupCosineAnnealingLR':
        sched_class = LinearWarmupCosineAnnealingLR
    else:
        sched_class = import_with_str('torch.optim.lr_scheduler', sched_name)
    scheduler = sched_class(optimizer, **sched_params)
    return scheduler

def create_tranform(transform_name, transform_params):
    if transform_params is None:
        transform_params = {}
    transform_class = import_with_str('datasets.transforms', transform_name)
    transform = transform_class(**transform_params)
    return transform

def create_dataset(dataset_name, dataset_params, pipeline):
    if dataset_params is None:
        dataset_params = {}
    dataset_class = import_with_str('datasets', dataset_name)
    dataset = dataset_class(pipeline=pipeline, **dataset_params)
    collate_fn = dataset_class.collate_fn
    return dataset, collate_fn

import torch
from torch.utils.data.dataloader import DataLoader

def create_dataloader(config):
    if config['dataset_info']['name'] == 'bible':
        from i_abstract_structure.dataset.dataset import bible as dataset_class
    else:
        raise ValueError('Invalid dataset name, currently supported [ bible ]')
    #
    data_path = config['dataset_info']['path']
    #
    train_object = dataset_class(
        path = data_path,
        cfg = config,
        seq_len = config['dataset_info']['seq_len'],
        mode = 'train',
    )
    train_loader = DataLoader(
        train_object,
        batch_size = config['dataset_info']['batch_size'],
        mode = 'train',
        # shuffle = True
        # num_workers = num_workers
    )
    #
    val_object = dataset_class(
        path = data_path,
        cfg = config,
        seq_len = config['dataset_info']['seq_len'],
        mode = 'val'
    )
    val_loader = DataLoader(
        val_object,
        batch_size = config['dataset_info']['batch_size'],
        mode = 'val'
    )
    return train_loader, val_loader


import torch
from torch.utils.data.dataloader import DataLoader

def create_dataloader(config):
    if config['name'] == 'bible':
        from i_abstract_structure.dataset.dataset import bible_dataset as dataset_class
    else:
        raise ValueError('Invalid dataset name, currently supported [ bible ]')
    #
    data_path = config['path']
    #
    train_object = dataset_class(
        cfg = config,
        seq_len = config['seq_len'],
        mode = 'train',
    )
    train_loader = DataLoader(
        train_object,
        batch_size = config['batch_size'],
        mode = 'train',
        # shuffle = True
        # num_workers = num_workers
    )
    #
    val_object = dataset_class(
        cfg = config,
        seq_len = config['seq_len'],
        mode = 'val'
    )
    val_loader = DataLoader(
        val_object,
        batch_size = config['batch_size'],
        mode = 'val'
    )
    return train_loader, val_loader

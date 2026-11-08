from .oct_transforms import oct_data_transforms
from .xchest_transforms import basic_xchest_transform
from .fundus_transforms import data_transforms, simple_transform

from utils.func import mean_and_std, print_dataset_info
from .dataset import FundusDataset, RSNADataset, OCTDataset

def generate_dataset(cfg):
    dset = cfg.base.dataset
    aug = cfg.data.augmentation
    if cfg.data.mean == 'auto' or cfg.data.std == 'auto':
        mean, std = auto_statistics(cfg)        
        cfg.data.mean = mean  
        cfg.data.std = std     

    if dset == 'Fundus':
        train_transform, test_transform = data_transforms(cfg) 
        datasets = generate_kaggle_dataset(cfg, train_transform, test_transform)
    elif dset == 'RSNA':
        train_transform, test_transform = basic_xchest_transform(cfg)
        datasets = generate_xchest_dataset(cfg, train_transform, test_transform)  
    elif dset == 'Retinal_OCT':   
           train_transform, test_transform = oct_data_transforms(cfg)
           datasets = generate_OCT_dataset(cfg, train_transform, test_transform) 
    else:
        raise ArgumentError(f'Dataset not implemented: {cfg.base.dataset}')

    print_dataset_info(datasets)
    return datasets

def auto_statistics(cfg):
    input_size = cfg.data.input_size,
    batch_size = cfg.train.batch_size,
    num_workers = cfg.train.num_workers
    print('Calculating mean and std of training set for data normalization.')
    transform = simple_transform(input_size)

    train_dataset = FundusDataset(cfg, transform=transform)

    return mean_and_std(train_dataset, batch_size, num_workers)


def generate_kaggle_dataset(cfg, train_transform, test_transform):          
    dset_train = FundusDataset(cfg, transform=train_transform)
    dset_val = FundusDataset(cfg, train=False, transform=test_transform)
    dset_test = FundusDataset(cfg, train=False, test=True, transform=test_transform)
    return dset_train, dset_test, dset_val

def generate_xchest_dataset(cfg, train_transform, test_transform):          
    dset_train = RSNADataset(cfg, transform=train_transform)
    dset_val = RSNADataset(cfg, train=False, transform=test_transform)
    dset_test = RSNADataset(cfg, train=False, test=True, transform=test_transform)
    return dset_train, dset_test, dset_val

def generate_OCT_dataset(cfg, train_transform, test_transform):                
    dset_train = OCTDataset(cfg, transform=train_transform)
    dset_val = OCTDataset(cfg, train=False, transform=test_transform)
    dset_test = OCTDataset(cfg, train=False, test=True, transform=test_transform)
    return dset_train, dset_test, dset_val
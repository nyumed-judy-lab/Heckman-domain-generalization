import torch
import typing
import random
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

# in our repo
from utils_datasets.samplers import configure_train_sampler


def dataloaders(args, dataset):
    # Get datasets. Note that we use ID validation.
    train_sets: typing.List[torch.utils.data.Dataset] = dataset.get_train_data(as_dict=False)
    id_validation_sets: typing.List[torch.utils.data.Dataset] = dataset.get_id_validation_data(as_dict=False)
    ood_validation_sets: typing.List[torch.utils.data.Dataset] = dataset.get_ood_validation_data(as_dict=False)  # or `None`
    test_sets: typing.List[torch.utils.data.Dataset] = dataset.get_test_data(as_dict=False)  # or `None`

    # torch dataset: training, validation, test
    train_set = ConcatDataset(train_sets)
    id_validation_set = ConcatDataset(id_validation_sets)
    ood_validation_set = ConcatDataset(ood_validation_sets)
    test_set = ConcatDataset(test_sets)

    # make dataloader
    train_sampler = configure_train_sampler(
        datasets=train_sets,
        use_domain=args.uniform_across_domains,
        use_target=args.uniform_across_targets,
    )

    # Instantiate train loader
    train_loader_configs = dict(
        batch_size= args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        num_workers=args.num_workers,
    )

    # Instantiate validation loader
    eval_loader_configs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=args.eval_prefetch_factor,
        num_workers=args.eval_num_workers,
    )

    # DATA LOADER
    train_loader = DataLoader(train_set, **train_loader_configs)
    id_valid_loader = DataLoader(id_validation_set, **eval_loader_configs)
    ood_valid_loader = DataLoader(ood_validation_set, **eval_loader_configs)
    test_loader = DataLoader(test_set, **eval_loader_configs)
    return train_loader, id_valid_loader, ood_valid_loader, test_loader




def sub_dataloaders(train_loader: torch.utils.data.DataLoader, 
                    id_valid_loader: torch.utils.data.DataLoader,
                    ood_valid_loader:  torch.utils.data.DataLoader,
                    test_loader: torch.utils.data.DataLoader,
                    subset_size: int = 3000,
                    ):
    train_loader, id_valid_loader, ood_valid_loader, test_loader
    if subset_size > len(train_loader.dataset):
        subset_size = int(len(train_loader.dataset)/2)
    elif subset_size > len(id_valid_loader.dataset):
        subset_size = int(len(id_valid_loader.dataset)/2) 
    elif subset_size > len(ood_valid_loader.dataset):
        subset_size = int(len(ood_valid_loader.dataset)/2) 
    elif subset_size > len(test_loader.dataset):
        subset_size = int(len(test_loader.dataset)/2)
    # subset_size = 2000  # number of samples in each subset
    sub_indices_train = random.sample(range(len(train_loader.dataset)), subset_size)
    sub_indices_valid_id = random.sample(range(len(id_valid_loader.dataset)), subset_size)
    sub_indices_valid_ood = random.sample(range(len(ood_valid_loader.dataset)), subset_size)
    sub_indices_test = random.sample(range(len(test_loader.dataset)), subset_size)

    sub_sampler_train = SubsetRandomSampler(sub_indices_train)
    sub_sampler_valid_id = SubsetRandomSampler(sub_indices_valid_id)
    sub_sampler_valid_ood = SubsetRandomSampler(sub_indices_valid_ood)
    sub_sampler_test = SubsetRandomSampler(sub_indices_test)

    sub_dataloader_train = DataLoader(train_loader.dataset, sampler=sub_sampler_train, batch_size=train_loader.batch_size, drop_last=True)
    sub_dataloader_valid_id = DataLoader(id_valid_loader.dataset, sampler=sub_sampler_valid_id, batch_size=id_valid_loader.batch_size, drop_last=True)
    sub_dataloader_valid_ood = DataLoader(ood_valid_loader.dataset, sampler=sub_sampler_valid_ood, batch_size=ood_valid_loader.batch_size, drop_last=True)
    sub_dataloader_test = DataLoader(test_loader.dataset, sampler=sub_sampler_test, batch_size=test_loader.batch_size, drop_last=True)
    
    return sub_dataloader_train, sub_dataloader_valid_id, sub_dataloader_valid_ood, sub_dataloader_test

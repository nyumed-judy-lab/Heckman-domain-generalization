import typing
import argparse
import torch
from utils_datasets.wilds_ import WildsCamelyonDataset
from utils_datasets.wilds_ import WildsIWildCamDataset
from utils_datasets.wilds_ import WildsRxRx1Dataset
from utils_datasets.wilds_ import WildsCivilCommentsDataset
# from datasets.wilds_ import WildsPovertyMapDataset, PovertyMapDataModule
from utils_datasets.domainbed_ import ColoredMNIST
from utils_datasets.domainbed_ import PACS, VLCS, DomainVLCS


def args_cameloyn17(args, experiment_name):
    args.data = 'camelyon17'
    args.backbone = 'densenet121'
    args.experiment_name = experiment_name
    args.batch_size = 32
    args.eval_batch_size = 32
    args.epochs = 5
    args.optimizer = 'adam'
    args.lr = 0.00001
    args.weight_decay = 0.0
    args.pretrained = True ###################################
    args.device = 'cuda'
    return args

def args_cameloyn17_outcome(args, experiment_name):
    args.data = 'camelyon17'
    args.backbone = 'densenet121'
    args.experiment_name = experiment_name
    args.batch_size = 32
    args.eval_batch_size = 32
    args.epochs = 5
    args.optimizer = 'sgd'
    args.lr = 0.001
    args.weight_decay = 0.00001 
    args.pretrained = True 
    # args.corr_optimizer = 'adam' 
    # args.corr_lr = 0.01 
    # args.corr_weight_decay = 0.0 
    # args.freeze_selection_encoder = True 
    # args.freeze_selection_head = True 
    args.device = 'cuda'
    return args

def args_poverty(args, experiment_name):
    args.backbone = 'resnet18'
    args.batch_size = 64
    args.experiment_name = experiment_name

    args.eval_batch_size = 64
    args.epochs = 10
    args.lr = 0.00001
    args.weight_decay = 0.0
    args.optimizer = 'adam'
    args.augmentation = True
    
    args.device = 'cuda'
    return args

def args_poverty_outcome(args, experiment_name):
    args.backbone = 'resnet18'
    args.batch_size = 64
    args.experiment_name = experiment_name

    args.eval_batch_size = 64
    args.epochs = 10
    args.lr = 0.001
    args.weight_decay = 0.00001
    args.optimizer = 'adam'
    args.augmentation = True
    # args.corr_optimizer = 'adam' ###################################
    # args.corr_lr = 0.01 ###################################
    # args.corr_weight_decay = 0.0 ###################################
    # args.freeze_selection_encoder = True ###################################
    # args.freeze_selection_head = True ###################################
    args.device = 'cuda'
    return args

def parse_arguments(name: str = "HeckmanDG") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=name, add_help=True)
    # name: str = "HeckmanDG pre-training"
    # parser = argparse.ArgumentParser(description=name, add_help=True)
    # experiment_name_list = ['HeckmanDG_pre-training_plain','HeckmanDG_pre-training_cv','HeckmanDG_pre-training_ece']

    parser.add_argument('--data', type=str, default='camelyon17', required=False, 
                        choices=('camelyon17', 'camelyon17_ece', 'poverty', 'rxrx1', 'iwildcam', 'civilcomments', 'pacs', 'vlcs', 'vlcs_ood',), help='')
    parser.add_argument('--experiment_name', type=str, default='plain', required=False, #True
                        choices=('plain','cv','ece'), help='')
    parser.add_argument('--backbone', type=str, default='resnet18', required=False, choices=('cnn', 'resnet18', 'resnet50', 'resnet101', 'densenet121', 'distilbert-base-uncased'), help='')
    parser.add_argument('--device', type=str, default='cuda', help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--cv', type=int, default=None)
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained weights (i.e., ImageNet) (default: False)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adam', 'adamw', 'lars', ), help='Optimizer (default: sgd)')
    parser.add_argument('--lr', type=float, default=0.01, help='Base learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay factor (default: 0.001')
    parser.add_argument('--epochs', type=int, default=30, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--augmentation', action='store_true', help='Apply augmentation to inputs (default: False)')
    
    ##################################################
    parser.add_argument('--pretrained_model_file', type=str, default=None, help='Path to checkpoint file from which weights are loaded (default: None)')
    # parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},help='Data-specific keyword arguments passed as key1=value key2=value2 ...')
    parser.add_argument('--eval_batch_size', type=int, default=512, help='')
    parser.add_argument('--randaugment', action='store_true', help='Apply RandAugment (default: False)')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze weights of encoder (default: False)')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Binary label smoothing factor (default: 0.0)')
    parser.add_argument('--focal', action='store_true', help='Enable focal loss weights. (default: False)')
    parser.add_argument('--scheduler', type=str, default=None, help='')
    parser.add_argument('--scheduler_lr_warmup', type=int, default=10, help='')
    parser.add_argument('--scheduler_min_lr', type=float, default=0., help='')
    parser.add_argument('--uniform_across_domains', action='store_true', help='')
    parser.add_argument('--uniform_across_targets', action='store_true', help='')
    parser.add_argument('--num_workers', type=int, default=12, help='')
    parser.add_argument('--eval_num_workers', type=int, default=12, help='')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='')
    parser.add_argument('--eval_prefetch_factor', type=int, default=4, help='')
    parser.add_argument('--save_every', type=int, default=1, help='')
    parser.add_argument('--early_stopping', type=int, default=None, help='')
    ##################################################
    args = parser.parse_args()
    # setattr(args, 'hash', create_hash())
    return args

def DatasetImporter(defaults, args):
    # instantiate dataset
    if args.data == 'camelyon17':
        dataset = WildsCamelyonDataset(
            root=defaults.root,
            train_domains=defaults.train_domains,
            validation_domains=defaults.validation_domains,
            test_domains=defaults.test_domains,
            reserve_id_validation=True,
            in_memory=defaults.load_data_in_memory,
        )

    elif args.data == 'camelyon17_ece':
        dataset = WildsCamelyonDataset(
            root=defaults.root,
            train_domains=defaults.train_domains,
            validation_domains=defaults.validation_domains,
            test_domains=defaults.test_domains,
            reserve_id_validation=True,
            in_memory=defaults.load_data_in_memory,
        )

    elif args.data == 'vlcs':
        dataset = VLCS(
            root=defaults.root,
            train_environments=defaults.train_environments,
            test_environments=defaults.test_environments,
            holdout_fraction=defaults.holdout_fraction,
            download=False,
        )
        setattr(defaults, 'train_domains', dataset.train_domains)
        setattr(defaults, 'test_domains', dataset.test_domains)

    elif args.data == 'vlcs_ood':
        
        dataset = DomainVLCS(
            root=defaults.root,
            train_domains=defaults.train_domains,
            validation_domains=defaults.validation_domains,
            test_domains=defaults.test_domains,
            holdout_fraction=defaults.holdout_fraction,            
            # in_memory=defaults.load_data_in_memory,
        )

    elif args.data == 'poverty':
        dataset = WildsPovertyMapDataset(
            root=defaults.root,
            # reserve_id_validation=True,
            # fold=defaults.fold,
            # in_memory=defaults.load_data_in_memory,
        )
        setattr(defaults, 'train_domains', dataset.train_domains)
        setattr(defaults, 'validation_domains', dataset.validation_domains)
        setattr(defaults, 'test_domains', dataset.test_domains)
    elif args.data == 'poverty_ece':
        dataset = WildsPovertyMapDataset(
            root=defaults.root,
            # reserve_id_validation=True,
            # fold=defaults.fold,
            # in_memory=defaults.load_data_in_memory,
        )
        setattr(defaults, 'train_domains', dataset.train_domains)
        setattr(defaults, 'validation_domains', dataset.validation_domains)
        setattr(defaults, 'test_domains', dataset.test_domains)
                
    # elif args.data == 'poverty':
    #     dataset = PovertyMapDataModule(
    #         root=defaults.root,
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #         prefetch_factor=args.prefetch_factor,
    #         )
    #     setattr(defaults, 'train_domains', dataset.train_domains)
    #     setattr(defaults, 'validation_domains', dataset.validation_domains)
    #     setattr(defaults, 'test_domains', dataset.test_domains)

    elif args.data == 'rxrx1':

        dataset = WildsRxRx1Dataset(root=defaults.root, reserve_id_validation=True)
        setattr(defaults, 'train_domains', dataset.train_domains)
        setattr(defaults, 'validation_domains', dataset.validation_domains)
        setattr(defaults, 'test_domains', dataset.test_domains)
    
    elif args.data == 'iwildcam':
    
        dataset = WildsIWildCamDataset(root=defaults.root)
        setattr(defaults, 'train_domains', dataset.train_domains)
        setattr(defaults, 'validation_domains', dataset.validation_domains)
        setattr(defaults, 'test_domains', dataset.test_domains)
    
    elif args.data == 'civilcomments':
        
        dataset = WildsCivilCommentsDataset(
            root=defaults.root,
            model=args.backbone,
            exclude_not_mentioned=defaults.exclude_not_mentioned,
        )
        setattr(defaults, 'train_domains', dataset.train_domains)
        setattr(defaults, 'validation_domains', dataset.validation_domains)
        setattr(defaults, 'test_domains', dataset.test_domains)
    
    elif args.data == 'cmnist':

        dataset = ColoredMNIST(
            train_domains=defaults.train_domains,
            test_domains=defaults.test_domains,
        )
    
    elif args.data == 'fmow':
        raise NotImplementedError

    else:
        raise ValueError
    return dataset

def load_checkpoint(self,
                    path: str,
                    encoder_keys: typing.Iterable[str] = ['encoder', 'selection_encoder'],
                    head_keys: typing.Iterable[str] = ['selection_head', ],
                    load_optimizer: bool = False,
                    load_scheduler: bool = False, ) -> None:

    ckpt = torch.load(path)
    self.logger.info(f"Loading weights from: {path}")

    # load encoder weights
    is_enc_loaded: bool = False
    for key in encoder_keys:
        try:
            self.selection_encoder.load_state_dict(ckpt[key])
            self.logger.info(f"Loaded encoder weights using key = `{key}`")
            is_enc_loaded = True
            break
        except KeyError as _:
            self.logger.info(f"Invalid key: `{key}`. Trying next key.")
            continue
    if not is_enc_loaded:
        self.logger.info(f"Failed to load encoder weights using keys from {encoder_keys}")
    
    # load head weights
    is_head_loaded: bool = False
    for key in head_keys:
        try:
            self.selection_head.load_state_dict(ckpt[key])
            self.logger.info(f"Loaded head weights using key = `{key}`")
            is_head_loaded = True
            break
        except KeyError as _:
            self.logger.info(f"Invalid key: `{key}`. Trying next key.")
            continue
    if not is_head_loaded:
        self.logger.info(f"Failed to load head weights using keys from {head_keys}")

    if load_optimizer:
        self.optimizer.load_state_dict(ckpt['optimizer'])
    
    if load_scheduler:
        if self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            except KeyError as _:
                pass


'''
# previous version of arguments
def parse_arguments(name: str = "HeckmanDG") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=name, add_help=True)
    # name: str = "HeckmanDG pre-training"
    # parser = argparse.ArgumentParser(description=name, add_help=True)
    # experiment_name_list = ['HeckmanDG_pre-training_plain','HeckmanDG_pre-training_cv','HeckmanDG_pre-training_ece']

    parser.add_argument('--data', type=str, default='camelyon17', required=False, 
                        choices=('camelyon17', 'camelyon17_ece', 'poverty', 'rxrx1', 'iwildcam', 'civilcomments', 'pacs', 'vlcs', 'vlcs_ood',), help='')
    parser.add_argument('--experiment_name', type=str, default='plain', required=False, #True
                        choices=('plain','cv','ece'), help='')
    parser.add_argument('--backbone', type=str, default='resnet18', required=False, choices=('cnn', 'resnet18', 'resnet50', 'resnet101', 'densenet121', 'distilbert-base-uncased'), help='')
    parser.add_argument('--device', type=str, default='cuda', help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--cv', type=int, default=None)
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained weights (i.e., ImageNet) (default: False)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adam', 'adamw', 'lars', ), help='Optimizer (default: sgd)')
    parser.add_argument('--lr', type=float, default=0.01, help='Base learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay factor (default: 0.001')
    parser.add_argument('--epochs', type=int, default=30, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--augmentation', action='store_true', help='Apply augmentation to inputs (default: False)')
    
    ##################################################
    parser.add_argument('--pretrained_model_file', type=str, default=None, help='Path to checkpoint file from which weights are loaded (default: None)')
    # parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},help='Data-specific keyword arguments passed as key1=value key2=value2 ...')
    parser.add_argument('--eval_batch_size', type=int, default=512, help='')
    parser.add_argument('--randaugment', action='store_true', help='Apply RandAugment (default: False)')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze weights of encoder (default: False)')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Binary label smoothing factor (default: 0.0)')
    parser.add_argument('--focal', action='store_true', help='Enable focal loss weights. (default: False)')
    parser.add_argument('--scheduler', type=str, default=None, help='')
    parser.add_argument('--scheduler_lr_warmup', type=int, default=10, help='')
    parser.add_argument('--scheduler_min_lr', type=float, default=0., help='')
    parser.add_argument('--uniform_across_domains', action='store_true', help='')
    parser.add_argument('--uniform_across_targets', action='store_true', help='')
    parser.add_argument('--num_workers', type=int, default=12, help='')
    parser.add_argument('--eval_num_workers', type=int, default=12, help='')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='')
    parser.add_argument('--eval_prefetch_factor', type=int, default=4, help='')
    parser.add_argument('--save_every', type=int, default=1, help='')
    parser.add_argument('--early_stopping', type=int, default=None, help='')
    ##################################################
    args = parser.parse_args()
    # setattr(args, 'hash', create_hash())
    return args


def parse_arguments_outcome(name: str = "HeckmanDG") -> argparse.Namespace:
    """Command-line arguments for HeckmanDG-1 training."""

    parser = argparse.ArgumentParser(description=name, add_help=True)
    
    # ...
    parser.add_argument('--data', type=str, default='camelyon17', required=False, 
                        choices=('camelyon17', 'camelyon17_ece', 'poverty', 'rxrx1', 'iwildcam', 'civilcomments', 'pacs', 'vlcs', 'vlcs_ood',), help='')
    parser.add_argument('--experiment_name', type=str, default='plain', required=False, #True
                        choices=('plain','cv','ece'), help='')

    parser.add_argument('--backbone', type=str, default='densenet121', choices=['resnet18', 'resnet50', 'resnet101', 'densenet121', 'distilbert-base-uncased', ], help='')
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--epochs', type=int, default=30,help='')
    parser.add_argument('--batch_size', type=int, default=256,help='')
    parser.add_argument('--eval_batch_size', type=int, default=512,help='')

    # parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={}, help='Data-specific keyword arguments passed as key1=value key2=value2 ...')
    parser.add_argument('--pretrained', action='store_true', help='Load ImageNet-pretrained model (default: False)')
    parser.add_argument('--selection_pretrained_model_file', type=str, default=None, help='File to load weights for selection model (default: None)')
    parser.add_argument('--outcome_pretrained_model_file', type=str, default=None, help='File to load weights for outcome model (default: None)')                        
    parser.add_argument('--optimizer', type=str, default='sgd',choices=['sgd', 'adam', 'adamw', ],help='')
    parser.add_argument('--lr', type=float, default=1e-3,help='base learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,help='l2 weight decay factor (default: 0.0001)')

    parser.add_argument('--corr_optimizer', type=str, default='adam',choices=['sgd', 'adam', 'adamw', ],help='')
    parser.add_argument('--corr_lr', type=float, default=1e-3, help='')
    parser.add_argument('--corr_weight_decay', type=float, default=0.,  help='')

    # parser.add_argument('--freeze_selection_encoder', action='store_true',help='')
    # parser.add_argument('--freeze_selection_head', action='store_true',help='')
    # parser.add_argument('--freeze_outcome_encoder', action='store_true',help='')

    parser.add_argument('--augmentation_selection', action='store_true', help='Apply data augmentation to inputs of selection model(s) (default: False)')
    parser.add_argument('--augmentation_outcome', action='store_true',help='Apply data augmentation to inputs of outcome model (default: False)')
    parser.add_argument('--randaugment_selection', action='store_true', help='Add RandAugment for selection.')
    parser.add_argument('--randaugment_outcome', action='store_true',help='Add RandAugment for outcome.')


    # ...
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--scheduler', type=str, default=None,help='')
    parser.add_argument('--scheduler_lr_warmup', type=int, default=10,help='')
    parser.add_argument('--scheduler_min_lr', type=float, default=0.,help='')


    # ...
    parser.add_argument('--corr_scheduler', type=str, default=None, help='')
    parser.add_argument('--corr_scheduler_lr_warmup', type=int, default=10,help='')
    parser.add_argument('--corr_scheduler_min_lr', type=float, default=0.,help='')

    # training
    
    # sampling
    parser.add_argument('--uniform_across_domains', action='store_true', help='Sample mini-batch uniformly with respect to domain (default: False)')
    parser.add_argument('--uniform_across_targets', action='store_true',help='Sample mini-batch uniformly with respect to target (default: False)')

    # cpu & gpu configurations
    parser.add_argument('--num_workers', type=int, default=4, help='')
    parser.add_argument('--eval_num_workers', type=int, default=8, help='')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='')
    parser.add_argument('--eval_prefetch_factor', type=int, default=4, help='')
    parser.add_argument('--save_every', type=int, default=1, help='')
    parser.add_argument('--early_stopping', type=int, default=None, help='')
    parser.add_argument('--sma_start_iter', type=int, default=-1, help='If -1, disables sma.')
    parser.add_argument('--calibrate', action='store_true', help='Calibrate selection model(s) (default: False)')
    

    args = parser.parse_args()
    assert isinstance(args.dataset_kwargs, dict)
    
    setattr(args, 'hash', create_hash())

    return args
'''

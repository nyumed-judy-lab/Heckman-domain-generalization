import pandas as pd
import typing
import argparse
import torch

from utils_datasets.base import MultipleDomainCollection
from utils_datasets.wilds_ import SingleCamelyon, WildsCamelyonDataset
from utils_datasets.wilds_ import SingleIWildCam, WildsIWildCamDataset
from utils_datasets.wilds_ import SingleRxRx1, WildsRxRx1Dataset
from utils_datasets.wilds_ import SinglePovertyMap, WildsPovertyMapDataset, WildsPovertyMapDataset_

def args_custom(args):
    args.data = ...
    args.backbone = ...
    args.batch_size = ...
    args.eval_batch_size = ...
    args.epochs = ...
    args.optimizer = ...
    args.lr = ...
    args.weight_decay = ...
    args.device = 'cuda'
    return args

def args_insight():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_domains', '-n', default=4, type=int)
    parser.add_argument('--seed', '-s', default=0, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--test_size', default=0.2, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--batchnorm', default=True, type=bool)
    parser.add_argument('--activation', default='ReLU', type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--eval_batch_size', default=500, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    
    args = parser.parse_args()
    args.root = './data/insight/static_new.feather'
    args.data = 'insight'
    args.data_type = 'tabular'
    args.backbone = 'mlp'
    # args.lr = 1e-2
    # args.weight_decay = 1e-2
    # args.dropout = 0.5
    # args.batchnorm = True
    # args.activation = 'ReLU'
    # args.batch_size = 500
    # args.eval_batch_size = 500
    # args.epochs = 150
    # args.optimizer = 'sgd'
    # args.device = 'cuda'
    return args

def args_cameloyn17(args):
    # Hyperparameters
    args.data = 'camelyon17'
    args.data_type = 'image'
    
    args.backbone = 'densenet121'
    args.batch_size = 32
    args.eval_batch_size = 32
    args.epochs = 5
    args.optimizer = 'sgd'
    args.lr = 0.001
    args.weight_decay = 0.00001 
    args.pretrained = True 
    args.augmentation = True ########## sanity check
    args.randaugment = True ##########     
    # args.corr_optimizer = 'adam' 
    # args.corr_lr = 0.01 
    # args.corr_weight_decay = 0.0 
    # args.freeze_selection_encoder = True 
    # args.freeze_selection_head = True 
    
    # Configuration
    args.root : str = './data/benchmark/wilds/camelyon17_v1.0'
    args.train_domains : typing.List[int] = [0, 3, 4]  # [0, 3, 4]
    args.validation_domains: typing.List[int] = [1]   # [1]
    args.test_domains: typing.List[int] = [2]         # [2]
    args.load_data_in_memory: int = 0
    args.model_selection: str = 'ood'
    args.model_selection_metric: str = 'accuracy' #TODO: ADD ECE
    args.num_classes: int = 2
    args.loss_type: str = 'binary'
    args.device = 'cuda'
    return args

def args_iwildcam(args):
    args.data_type = 'image'
    args.data = 'iwildcam'
    # Hyperparameters
    
    args.backbone = 'resnet50'
    args.epochs = 12
    args.batch_size = 16
    args.eval_batch_size = 16
    args.optimizer = 'sgd'
    args.lr = 0.001
    args.weight_decay = 0.00001
    args.pretrained = True 
    args.augmentation = True ########## 
    args.randaugment = True ########## 
    args.device = 'cuda'
    
    ###################################
    args.corr_optimizer = 'adam' 
    args.corr_lr = 0.01 
    args.corr_weight_decay = 0.0 
    ###################################
    
    # Configurations
    args.root: str = './data/benchmark/wilds/iwildcam_v2.0'
    args.model_selection: str = 'ood'
    args.model_selection_metric: str = 'f1'  # macro it is
    args.num_classes: int = 182
    args.loss_type: str = 'multiclass'
    
    # args.num_domains = 243
    
    return args

def args_poverty(args):
    args.data = 'poverty'
    args.data_type = 'image'
    # hyperparameters
    args.backbone = 'resnet18'
    args.epochs = 100
    args.batch_size = 64
    args.eval_batch_size = 64
    args.optimizer = 'sgd'
    args.lr = 0.001
    args.weight_decay = 0.00001
    args.pretrained = False 
    args.augmentation = True ########## 
    args.randaugment = True ########## 
    args.device = 'cuda'
    # args.corr_optimizer = 'adam' ###################################
    # args.corr_lr = 0.01 ###################################
    # args.corr_weight_decay = 0.0 ###################################
    
    # configuration
    args.root: str = './data/benchmark/wilds/poverty_v1.1'
    args.fold: str = 'A'  # [A, B, C, D, E]
    args.use_area: bool = False
    args.load_data_in_memory: int = 0
    args.model_selection: str = 'ood'
    args.model_selection_metric: str = 'pearson'
    args.num_classes: int = None  # type: ignore
    args.loss_type: str = 'regression'
    
    return args

def args_rxrx1(args):
    args.data = 'rxrx1'
    args.data_type = 'image'
    
    # Hypermarameters
    args.backbone = 'resnet50'
    args.epochs = 10
    args.batch_size = 75
    args.eval_batch_size = 75
    args.optimizer = 'sgd'
    args.lr = 0.001
    args.weight_decay = 0.00001
    args.pretrained = True 
    args.augmentation = True ########## 
    args.randaugment = True ########## 
    args.device = 'cuda'
    
    # args.corr_optimizer = 'adam' ###################################
    # args.corr_lr = 0.01 ###################################
    # args.corr_weight_decay = 0.0 ###################################
    
    # Configurations
    args.root: str = './data/benchmark/wilds/rxrx1_v1.0'
    args.load_data_in_memory: int = 0  # 12 or 0
    args.model_selection: str = 'ood'
    args.model_selection_metric: str = 'accuracy'    
    args.num_classes: int = 1139
    args.loss_type: str = 'multiclass'
    # args.num_domains = 33
    return args

def parse_arguments(name: str = "HeckmanDG") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=name, add_help=True)
    parser.add_argument('--data', type=str, default='camelyon17', required=False, choices=('insight', 'camelyon17', 'poverty', 'rxrx1', 'iwildcam', 'civilcomments', 'pacs', 'vlcs'), help='')
    parser.add_argument('--experiment_name', type=str, default='Heckman DG', required=False, help='')
    parser.add_argument('--backbone', type=str, default='resnet18', required=False, choices=('mlp', 'cnn', 'resnet18', 'resnet50', 'resnet101', 'densenet121', 'distilbert-base-uncased'), help='')
    parser.add_argument('--device', type=str, default='cuda', help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--cv', type=int, default=None)
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained weights (i.e., ImageNet) (default: False)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adam', 'adamw', 'lars', ), help='Optimizer (default: sgd)')
    parser.add_argument('--lr', type=float, default=0.01, help='Base learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay factor (default: 0.001')
    parser.add_argument('--epochs', type=int, default=30, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--eval_batch_size', type=int, default=512, help='')
    parser.add_argument('--augmentation', action='store_true', help='Apply augmentation to inputs (default: False)')
    parser.add_argument('--randaugment', action='store_true', help='Apply RandAugment (default: False)')
    
    ##################################################
    parser.add_argument('--num_workers', type=int, default=12, help='')
    parser.add_argument('--eval_num_workers', type=int, default=12, help='')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='')
    parser.add_argument('--eval_prefetch_factor', type=int, default=4, help='')
    parser.add_argument('--save_every', type=int, default=1, help='')
    parser.add_argument('--early_stopping', type=int, default=None, help='')
    parser.add_argument('--scheduler', type=str, default=None, help='')
    parser.add_argument('--scheduler_lr_warmup', type=int, default=10, help='')
    parser.add_argument('--scheduler_min_lr', type=float, default=0., help='')
    parser.add_argument('--uniform_across_domains', action='store_true', help='')
    parser.add_argument('--uniform_across_targets', action='store_true', help='')
    ##################################################
    # parser.add_argument('--pretrained_model_file', type=str, default=None, help='Path to checkpoint file from which weights are loaded (default: None)')
    # parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},help='Data-specific keyword arguments passed as key1=value key2=value2 ...')
    # parser.add_argument('--freeze_encoder', action='store_true', help='Freeze weights of encoder (default: False)')
    # parser.add_argument('--label_smoothing', type=float, default=0.0, help='Binary label smoothing factor (default: 0.0)')
    # parser.add_argument('--focal', action='store_true', help='Enable focal loss weights. (default: False)')
    args = parser.parse_args()
    # setattr(args, 'hash', create_hash())
    return args

def get_countires(fold):
    if fold == 'A':
        train_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17A['train']
        validation_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17A['ood_val']
        test_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17A['ood_test']
    elif fold == 'B':
        train_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17B['train']
        validation_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17B['ood_val']
        test_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17B['ood_test']
    elif fold == 'C':
        train_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17C['train']
        validation_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17C['ood_val']
        test_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17C['ood_test']
    elif fold == 'D':
        train_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17D['train']
        validation_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17D['ood_val']
        test_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17D['ood_test']
    elif fold == 'E':
        train_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17E['train']
        validation_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17E['ood_val']
        test_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17E['ood_test']
    else:
        print("choose in ['A','B','C','D','E']")
    return train_countries, validation_countries, test_countries

def DatasetImporter(args):
    # instantiate dataset
    if args.data == 'camelyon17':
        dataset = WildsCamelyonDataset(
            root=args.root,
            train_domains=args.train_domains,
            validation_domains=args.validation_domains,
            test_domains=args.test_domains,
        )
        # setattr(args, 'train_domains', dataset.train_domains)
        # setattr(args, 'validation_domains', dataset.validation_domains)
        # setattr(args, 'test_domains', dataset.test_domains)
        setattr(args, 'num_domains', len(dataset.train_domains))


    elif args.data == 'poverty':
        # args.data = 'poverty' 
        
        dataset = WildsPovertyMapDataset(
            root=args.root,
            train_domains = get_countires(args.fold)[0],
            validation_domains = get_countires(args.fold)[1],
            test_domains = get_countires(args.fold)[2]
        )
        '''
        dataset = WildsPovertyMapDataset_(
            args=args
        )
        dataset._train_datasets
        dataset._validation_datasets
        dataset._test_datasets
        '''
        setattr(args, 'train_domains', dataset.train_domains)
        setattr(args, 'validation_domains', dataset.validation_domains)
        setattr(args, 'test_domains', dataset.test_domains)
        setattr(args, 'num_domains', len(dataset.train_domains))
        
    elif args.data == 'rxrx1':
        # args.data = 'rxrx1'    
        dataset = WildsRxRx1Dataset(root=args.root, reserve_id_validation=True)
        setattr(args, 'train_domains', dataset.train_domains)
        setattr(args, 'validation_domains', dataset.validation_domains)
        setattr(args, 'test_domains', dataset.test_domains)
        setattr(args, 'num_domains', len(dataset.train_domains))
    
    elif args.data == 'iwildcam':
        # args.data = 'iwildcam'    
        dataset = WildsIWildCamDataset(root=args.root)
        setattr(args, 'train_domains', dataset.train_domains)
        setattr(args, 'validation_domains', dataset.validation_domains)
        setattr(args, 'test_domains', dataset.test_domains)
        setattr(args, 'num_domains', len(dataset.train_domains))
        
    elif args.data == 'insight':
        # args.data = 'insight'    
        dataset = pd.read_feather(args.root)
    else:
        raise ValueError
    return dataset
    
# def DatasetImporter(defaults, args):
#     # instantiate dataset
#     if args.data == 'camelyon17':
#         dataset = WildsCamelyonDataset(
#             root=defaults.root,
#             train_domains=defaults.train_domains,
#             validation_domains=defaults.validation_domains,
#             test_domains=defaults.test_domains,
#             reserve_id_validation=True,
#             in_memory=defaults.load_data_in_memory,
#         )
#     elif args.data == 'rxrx1':
    
#         dataset = WildsRxRx1Dataset(root=defaults.root, reserve_id_validation=True)
#         setattr(defaults, 'train_domains', dataset.train_domains)
#         setattr(defaults, 'validation_domains', dataset.validation_domains)
#         setattr(defaults, 'test_domains', dataset.test_domains)
    
#     elif args.data == 'iwildcam':
#         dataset = WildsIWildCamDataset(root=defaults.root)
#         setattr(defaults, 'train_domains', dataset.train_domains)
#         setattr(defaults, 'validation_domains', dataset.validation_domains)
#         setattr(defaults, 'test_domains', dataset.test_domains)

#     elif args.data == 'poverty':
#         dataset = WildsPovertyMapDataset(
#             root=defaults.root,
#             # reserve_id_validation=True,
#             # fold=defaults.fold,
#             # in_memory=defaults.load_data_in_memory,
#         )
#         setattr(defaults, 'train_domains', dataset.train_domains)
#         setattr(defaults, 'validation_domains', dataset.validation_domains)
#         setattr(defaults, 'test_domains', dataset.test_domains)
                
#     # elif args.data == 'poverty':
#     #     dataset = PovertyMapDataModule(
#     #         root=defaults.root,
#     #         batch_size=args.batch_size,
#     #         num_workers=args.num_workers,
#     #         prefetch_factor=args.prefetch_factor,
#     #         )
#     #     setattr(defaults, 'train_domains', dataset.train_domains)
#     #     setattr(defaults, 'validation_domains', dataset.validation_domains)
#     #     setattr(defaults, 'test_domains', dataset.test_domains)
    
#     else:
#         raise ValueError
#     return dataset

# def load_checkpoint(self,
#                     path: str,
#                     encoder_keys: typing.Iterable[str] = ['encoder', 'selection_encoder'],
#                     head_keys: typing.Iterable[str] = ['selection_head', ],
#                     load_optimizer: bool = False,
#                     load_scheduler: bool = False, ) -> None:

#     ckpt = torch.load(path)
#     self.logger.info(f"Loading weights from: {path}")

#     # load encoder weights
#     is_enc_loaded: bool = False
#     for key in encoder_keys:
#         try:
#             self.selection_encoder.load_state_dict(ckpt[key])
#             self.logger.info(f"Loaded encoder weights using key = `{key}`")
#             is_enc_loaded = True
#             break
#         except KeyError as _:
#             self.logger.info(f"Invalid key: `{key}`. Trying next key.")
#             continue
#     if not is_enc_loaded:
#         self.logger.info(f"Failed to load encoder weights using keys from {encoder_keys}")
    
#     # load head weights
#     is_head_loaded: bool = False
#     for key in head_keys:
#         try:
#             self.selection_head.load_state_dict(ckpt[key])
#             self.logger.info(f"Loaded head weights using key = `{key}`")
#             is_head_loaded = True
#             break
#         except KeyError as _:
#             self.logger.info(f"Invalid key: `{key}`. Trying next key.")
#             continue
#     if not is_head_loaded:
#         self.logger.info(f"Failed to load head weights using keys from {head_keys}")

#     if load_optimizer:
#         self.optimizer.load_state_dict(ckpt['optimizer'])
    
#     if load_scheduler:
#         if self.scheduler is not None:
#             try:
#                 self.scheduler.load_state_dict(ckpt['scheduler'])
#             except KeyError as _:
#                 pass


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


"""
    Default configurations for each dataset considered in this study.
"""
import typing


class DefaultConfig(object):
    pretrain_model_selection_metric: str = 'f1'
    def __init__(self):
        pass
class DefaultConfig_ece(object):
    pretrain_model_selection_metric: str = 'ece'
    def __init__(self):
        pass

class Camelyon17Defaults(DefaultConfig):
    def __init__(self):
        super(Camelyon17Defaults, self).__init__()

        self.root: str = './data/benchmark/wilds/camelyon17_v1.0'
        self.train_domains: typing.List[int] = [0, 3, 4]  # [0, 3, 4]
        self.validation_domains: typing.List[int] = [1]   # [1]
        self.test_domains: typing.List[int] = [2]         # [2]

        self.load_data_in_memory: int = 0

        self.model_selection: str = 'ood'
        self.model_selection_metric: str = 'accuracy' #TODO: ADD ECE

        self.num_classes: int = 2
        self.loss_type: str = 'binary'

class Camelyon17Defaults_ece(DefaultConfig_ece):
    def __init__(self):
        super(Camelyon17Defaults_ece, self).__init__()

        self.root: str = './data/benchmark/wilds/camelyon17_v1.0'
        self.train_domains: typing.List[int] = [0, 3, 4]  # [0, 3, 4]
        self.validation_domains: typing.List[int] = [1]   # [1]
        self.test_domains: typing.List[int] = [2]         # [2]

        self.load_data_in_memory: int = 0

        self.model_selection: str = 'ood'
        self.model_selection_metric: str = 'accuracy' #TODO: ADD ECE

        self.num_classes: int = 2
        self.loss_type: str = 'binary'

class PovertyMapDefaults(DefaultConfig):
    def __init__(self):
        super(PovertyMapDefaults, self).__init__()

        self.root: str = './data/benchmark/wilds/poverty_v1.1'
        self.fold: str = 'A'  # [A, B, C, D, E]
        
        self.use_area: bool = False
        self.load_data_in_memory: int = 0

        self.model_selection: str = 'ood'
        self.model_selection_metric: str = 'pearson'

        self.num_classes: int = None  # type: ignore
        self.loss_type: str = 'regression'


class PovertyMapDefaults_ece(DefaultConfig_ece):
    def __init__(self):
        super(PovertyMapDefaults_ece, self).__init__()

        self.root: str = './data/benchmark/wilds/poverty_v1.1'
        self.fold: str = 'A'  # [A, B, C, D, E]
        
        self.use_area: bool = False
        self.load_data_in_memory: int = 0

        self.model_selection: str = 'ood'
        self.model_selection_metric: str = 'pearson'

        self.num_classes: int = None  # type: ignore
        self.loss_type: str = 'regression'

class VLCSDefaults(DefaultConfig):
    def __init__(self):
        super(VLCSDefaults, self).__init__()

        self.root: str = './data/benchmark/domainbed/vlcs'

        self.train_environments: typing.List[str] = ['V', 'L', 'C']
        self.test_environments: typing.List[str] = ['S']

        self.model_selection: str = 'id'
        self.model_selection_metric: str = 'accuracy'

        self.holdout_fraction: float = 0.2

        self.num_classes: int = 5
        self.loss_type: str = 'multiclass'

class VLCSDefaults_OOD(DefaultConfig):
    def __init__(self):
        super(VLCSDefaults_OOD, self).__init__()

        self.root: str = './data/benchmark/domainbed/vlcs'

        # self.train_domains: typing.List[str] = ['V', 'L']
        # self.validation_domains: typing.List[str] = ['C']
        # self.test_domains: typing.List[str] = ['S']

        self.train_domains: typing.List[int] = [0, 1]
        self.validation_domains: typing.List[int] = [2]
        self.test_domains: typing.List[int] = [3]
        # self.load_data_in_memory: int = 0

        self.model_selection: str = 'ood'
        self.model_selection_metric: str = 'accuracy'

        self.holdout_fraction: float = 0.2

        self.num_classes: int = 5
        self.loss_type: str = 'multiclass'


class RotatedMNISTDefaults(DefaultConfig):
    def __init__(self):
        super(RotatedMNISTDefaults, self).__init__()
        
        self.root: str = './data/benchmark/rmnist'
        
        self.environments: typing.List[int] = [0, 15, 30, 45, 60, 75, 90]
        self.test_environment: int = 90
        self.ood_val_environment: int = 75
        self.num_classes: int = 10
        
        self.loss_type: str = 'multiclass'


class ColoredMNISTDefaults(DefaultConfig):
    def __init__(self):
        super(ColoredMNISTDefaults, self).__init__()

        self.root: str = './data/benchmark/mnist'  # not used
        
        self.train_domains: typing.List[int] = [0, 1]
        self.test_domains: typing.List[int] = [2]

        self.model_selection: str = 'id'
        self.model_selection_metric: str = 'accuracy'

        self.holdout_fraction: float = 0.2

        self.num_classes: int = 2
        self.loss_type: str = 'binary'


class PACSDefaults(DefaultConfig):
    def __init__(self):
        super(PACSDefaults, self).__init__()

        self.root: str = './data/benchmark/domainbed/pacs'

        self.train_environments: typing.List[str] = ['P', 'A', 'C']
        self.test_environments: typing.List[str] = ['S']

        self.model_selection: str = 'id'
        self.model_selection_metric: str = 'accuracy'

        self.holdout_fraction: float = 0.2

        self.num_classes: int = 7
        self.loss_type: str = 'multiclass'

class IWildCamDefaults(DefaultConfig):
    def __init__(self):
        super(IWildCamDefaults, self).__init__()
    
        self.root: str = './data/benchmark/wilds/iwildcam_v2.0'

        self.model_selection: str = 'ood'
        self.model_selection_metric: str = 'f1'  # macro it is

        self.num_classes: int = 182
        self.loss_type: str = 'multiclass'


class FMoWDefaults(DefaultConfig):
    def __init__(self):
        super(FMowDefaults, self).__init__()

        self.root: str = './data/benchmark/wilds/fmow_v1.0'  # TODO: check
    
        self.num_classes: int = 62                 # TODO: check
        self.loss_type: str = 'multiclass'

        self.model_selection: str = 'ood'
        self.model_selection_metric: str = 'accuracy_wg'  # worst region


# class PovertyMapDefaults(DefaultConfig):
#     def __init__(self):
#         super(PovertyMapDefaults, self).__init__()

#         self.root: str = './data/benchmark/wilds/poverty_v1.1'
#         self.fold: str = 'A'  # [A, B, C, D, E]
        
#         self.use_area: bool = False
#         self.load_data_in_memory: int = 0

#         self.model_selection: str = 'ood'
#         self.model_selection_metric: str = 'pearson'

#         self.num_classes: int = None  # type: ignore
#         self.loss_type: str = 'regression'


class CivilCommentsDefaults(DefaultConfig):
    def __init__(self):
        super(CivilCommentsDefaults, self).__init__()

        self.root: str = './data/benchmark/wilds/civilcomments_v1.0'

        self.exclude_not_mentioned: bool = False

        self.model_selection: str = 'id'
        self.model_selection_metric: str = 'accuracy_wg'

        self.num_classes: int = 2
        self.loss_type: str = 'binary'


class CivilCommentsDGDefaults(DefaultConfig):
    def __init__(self):
        super(CivilCommentsDGDefaults, self).__init__()

        self.root: str = './data/benchmark/wilds/civilcomments_v1.0'

        self.exclude_not_mentioned: bool = True

        self.model_selection: str = 'ood'
        self.model_selection_metric: str = 'accuracy'

        self.num_classes: int = 2
        self.loss_type: str = 'binary'


class CelebADefaults(DefaultConfig):
    def __init__(self):
        super(CelebADefaults, self).__init__()

        self.root: str = './data/benchmark/wilds/celebA_v1.0'
        self.model_selection: str = 'id'
        self.model_selection_metric: str = 'accuracy_wg'

        self.num_classes: int = 2
        self.loss_type: str = 'binary'
        self.train_domains: list = ['male', 'female']  # mind the order


class RxRx1Defaults(DefaultConfig):
    def __init__(self):
        super(RxRx1Defaults, self).__init__()

        self.root: str = './data/benchmark/wilds/rxrx1_v1.0'

        self.load_data_in_memory: int = 0  # 12 or 0

        self.model_selection: str = 'ood'
        self.model_selection_metric: str = 'accuracy'
        
        self.num_classes: int = 1139
        self.loss_type: str = 'multiclass'
        

DataDefaults: typing.Dict[str, DefaultConfig] = {
    'camelyon17': Camelyon17Defaults,
    'camelyon17_ece': Camelyon17Defaults_ece,
    'rxrx1': RxRx1Defaults,
    'iwildcam': IWildCamDefaults,
    'rmnist': RotatedMNISTDefaults,
    'cmnist': ColoredMNISTDefaults,
    'pacs': PACSDefaults,
    'vlcs': VLCSDefaults,
    'vlcs_ood': VLCSDefaults_OOD,
    'poverty': PovertyMapDefaults,
    'poverty_ece': PovertyMapDefaults_ece,
    'civilcomments': CivilCommentsDefaults,
    'celeba': CelebADefaults,
}

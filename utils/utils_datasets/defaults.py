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

class IWildCamDefaults(DefaultConfig):
    def __init__(self):
        super(IWildCamDefaults, self).__init__()
    
        self.root: str = './data/benchmark/wilds/iwildcam_v2.0'

        self.model_selection: str = 'ood'
        self.model_selection_metric: str = 'f1'  # macro it is

        self.num_classes: int = 182
        self.loss_type: str = 'multiclass'

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

class RxRx1Defaults(DefaultConfig):
    def __init__(self):
        super(RxRx1Defaults, self).__init__()

        self.root: str = './data/benchmark/wilds/rxrx1_v1.0'

        self.load_data_in_memory: int = 0  # 12 or 0

        self.model_selection: str = 'ood'
        self.model_selection_metric: str = 'accuracy'
        
        self.num_classes: int = 1139
        self.loss_type: str = 'multiclass'


class INSIGHTDefaults(DefaultConfig):
    def __init__(self):
        super(INSIGHTDefaults, self).__init__()

        self.root: str = './data/insight'

        self.model_selection: str = 'ood'
        self.model_selection_metric: str = 'accuracy'
        
        self.num_classes: int = 2
        self.loss_type: str = 'binary'        

DataDefaults: typing.Dict[str, DefaultConfig] = {
    'insight': INSIGHTDefaults,
    'camelyon17': Camelyon17Defaults,
    'rxrx1': RxRx1Defaults,
    'iwildcam': IWildCamDefaults,
    'poverty': PovertyMapDefaults,
    # 'camelyon17_ece': Camelyon17Defaults_ece,
    # 'rmnist': RotatedMNISTDefaults,
    # 'cmnist': ColoredMNISTDefaults,
    # 'pacs': PACSDefaults,
    # 'vlcs': VLCSDefaults,
    # 'vlcs_ood': VLCSDefaults_OOD,
    # 'poverty_ece': PovertyMapDefaults_ece,
    # 'civilcomments': CivilCommentsDefaults,
    # 'celeba': CelebADefaults,
}

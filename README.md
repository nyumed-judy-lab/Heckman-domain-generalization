# image-benchmark-domain-generalization
<!-- 
### ONE FILE for all datasets (and the readme.md file )
-->

This repository provides the PyTorch implementation of Heckman DG on WILDS benchmark data. We use the one-step optimization to train the Heckman DG model. In the Heckman DG model, the selection (g) and outcome (f) networks are composed of data-specific convolutional neural networks (CNN) structures recommended by [WILDS](https://proceedings.mlr.press/v139/koh21a) paper. In addition, we follow the hyperparameters of each data and model recommended by [HeckmanDG](https://openreview.net/forum?id=fk7RbGibe1) paper. Please note that the current repository provides only the code to run the experiment of Camelyon17 data. The codes for other data will be added. 

<!-- 
## References
Please refer to the following papers to set hyperparameters and reproduce experiments on the WILDS benchmark.

- [WILDS](https://proceedings.mlr.press/v139/koh21a) paper: Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., ... & Liang, P. (2021, July). Wilds: A benchmark of in-the-wild distribution shifts. In International Conference on Machine Learning (pp. 5637-5664). PMLR.

- [HeckmanDG](https://openreview.net/forum?id=fk7RbGibe1) paper: Kahng, H., Do, H., & Zhong, J. Domain Generalization via Heckman-type Selection Models. In The Eleventh International Conference on Learning Representations.
-->

## 1. Installation
Please see [requirements.txt](requirements.txt). It presents the names and versions of all libraries that we have to install before the implementation of this repository. Please note that we mainly use the Pytorch backend libraries as follows:
- torch==1.10.0
- torchaudio==0.10.0
- torchmetrics==0.11.1
- torchvision==0.11.0

```bash
# pip
pip install -r requirements.txt

# conda
conda install --file requirements.txt
```

## 2. Data Preparation
We then need to download benchmark image (**WILDS**) data. Please run the following code. 

``` bash
# Run 1.download_wilds_data.py

python 1.download_wilds_data.py --root_dir ./data/benchmark/wilds
```
### WILDS benchmark
WILDS benchmark includes four datasets; Camelyon 17, PovertyMap, iWildCam, and RxRx1. Below are the details of each data.
- Camelyon17: Binary (tumor) classification.
- PovertyMap: Regression (wealth index prediction).
- iWildCam: multiclass (animal species) classification.
- RxRx1: multiclass (genetic treatments) classification.

![image](https://user-images.githubusercontent.com/36376255/226856940-2cca2f56-abee-46fa-9ec9-f187c6ac290b.png)

## 3.Experiments
please go to [main_heckmandg.py](main_heckmandg.py) and run it as follows:

```bash
# Run Heckman DG on Camelyon17 data with (batch_size, 3, 96, 96) input image and binary outcome

python 2.run-cameloyon17-CNN-OneStep-HeckmanDG.py
```

### Brief introduction of the code: This code is composed of the following 4 steps.

**1. Experiment Settings**
- Here, we set the name of data (camelyon17) and data-specific hyperparameters. The recommended data-specific hyperparameters are already set, so if you want to see results with other settings, please modify the **args** variable in the [2.run-cameloyon17-CNN-OneStep-HeckmanDG.py](2.run-cameloyon17-CNN-OneStep-HeckmanDG.py). 

```python
from utils_datasets.defaults import DataDefaults
from utils.argparser import DatasetImporter, parse_arguments, args_cameloyn17_outcome
from utils.dataloader import dataloaders, sub_dataloaders

data_name = 'camelyon17'
data_type = 'image'
experiment_name = 'Heckman DG Benchmark'

args = parse_arguments(experiment_name) # basic arguments for image data
args = args_cameloyn17_outcome(args, experiment_name) # data-specific arguments 
# DataDefaults: has data-specific hyperparameters
defaults = DataDefaults[args.data]() 
args.num_domains = len(defaults.train_domains)
args.num_classes = defaults.num_classes
args.loss_type = defaults.loss_type
args.train_domains = defaults.train_domains
args.validation_domains = defaults.validation_domains
args.test_domains = defaults.test_domains

fix_random_seed(args.seed)    
```

**2. Data Preparation**
- The WILDS data basically require a large computing memory for the training step. If you want to test this code with the smaller size of data (subsets of the original data), please add (or uncomment) the following code at lines 50 to 54.

```python
# DatasetImporter: put DataDefaults into DatasetImporter to get the dataset
dataset = DatasetImporter(defaults, args)

# (1) run the experiment with all data to test the implementation of HeckmanDG (take large amount of memory)
train_loader, valid_loader, test_loader = dataloaders(args, dataset)

'''
# (2) run the experiment with subset of data to test the implementation of HeckmanDG (take small amount of memory)
if True:
    train_loader, valid_loader, test_loader = sub_dataloaders(train_loader, valid_loader, test_loader)
'''
```

**3. HeckmanDG**
- Here, we initialize the network (CNN) and optimizer and run the Heckman DG model.

```python
from networks import SeparatedHeckmanNetwork, SeparatedHeckmanNetworkCNN # 
from models import HeckmanDGBinaryClassifier, HeckmanDGBinaryClassifierCNN # 

if data_type == 'tabular':
    # len(train_loader.dataset['x'].shape)>4
    network = SeparatedHeckmanNetwork(args)
    optimizer = partial(torch.optim.SGD, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[2, 4], gamma=.1)
    model = HeckmanDGBinaryClassifier(args, network, optimizer, scheduler)
    model.fit(train_loader, valid_loader)
elif data_type == 'image':
    # len(train_loader.dataset['x'].shape)<4
    network = SeparatedHeckmanNetworkCNN(args)
    optimizer = partial(torch.optim.SGD, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[2, 4], gamma=.1)
    model = HeckmanDGBinaryClassifierCNN(args, network, optimizer, scheduler)
    model.fit(train_loader, valid_loader)
```

**4. Result Analysis**
- The results of this code are as follows:
  - plots of the training loss [learning curve](results/plots/HeckmanDG_camelyon17_loss.pdf)
  - plots of the probits [histogram](results/plots/HeckmanDG_camelyon17_probits.pdf)
  - AUC scores [prediction results](results/prediction/HeckmanDG_camelyon17.csv)

```python
# plots: loss, probits
from utils.plots import plots_loss, plots_probit
domain_color_map = {
    0: 'orange',
    1: 'slateblue',
    2: 'navy',
    3: 'crimson',
    4: 'darkgreen',
}

plots_loss(model, args, domain_color_map, path=f"./results/plots/HeckmanDG_{args.data}_loss.pdf")
probits, labels = model.get_selection_probit(train_loader)
plots_probit(probits, labels, args, domain_color_map, path=f"./results/plots/HeckmanDG_{args.data}_probits.pdf")

# prediction results
res_tr = []
res_vl = []
res_ts = []
for b, batch in enumerate(train_loader):
    print(f'train_loader {b} batch / {len(train_loader)}')
    y_true = batch['y']
    y_pred = model.predict_proba(batch)
    try:
        score = roc_auc_score(y_true, y_pred)
        res_tr.append(score)
    except ValueError:
        pass

for b, batch in enumerate(valid_loader):
    print(f'valid_loader {b} batch / {len(valid_loader)}')
    y_true = batch['y']
    y_pred = model.predict_proba(batch)
    try:
        score = roc_auc_score(y_true, y_pred)
        res_vl.append(score)
    except ValueError:
        pass
for b, batch in enumerate(test_loader):
    print(f'{b} batch / {len(test_loader)}')
    y_true = batch['y']
    y_pred = model.predict_proba(batch)
    try:
        score = roc_auc_score(y_true, y_pred)
        res_ts.append(score)
    except ValueError:
        pass
    
res_tr_mean = pd.DataFrame(res_tr).mean()
res_vl_mean = pd.DataFrame(res_vl).mean()
res_ts_mean = pd.DataFrame(res_ts).mean()

results = pd.concat([pd.DataFrame([args.data]), res_tr_mean, res_vl_mean, res_ts_mean], axis=1)
results.columns = ['data', 'train', 'valid', 'test']
print(results)
results.to_csv(f'./results/prediction/HeckmanDG_{args.data}.csv')
```

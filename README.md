# image-benchmark-domain-generalization

This repository provides the PyTorch implementation of Heckman DG on WILDS benchmark data. We use the one-step optimization to train the Heckman DG model. In the Heckman DG model, the selection (g) and outcome (f) networks are composed of data-specific convolutional neural networks (CNN) structures recommended by [WILDS](https://proceedings.mlr.press/v139/koh21a) paper. In addition, we follow the hyperparameters of each data and model recommended by [HeckmanDG](https://openreview.net/forum?id=fk7RbGibe1) paper. Please note that the current repository provides only the code to run the experiment of Camelyon17 data. The codes for other data will be added. 

## References
Please refer to the following papers to set hyperparameters and reproduce experiments on the WILDS benchmark.

- [WILDS](https://proceedings.mlr.press/v139/koh21a) paper: Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., ... & Liang, P. (2021, July). Wilds: A benchmark of in-the-wild distribution shifts. In International Conference on Machine Learning (pp. 5637-5664). PMLR.

- [HeckmanDG](https://openreview.net/forum?id=fk7RbGibe1) paper: Kahng, H., Do, H., & Zhong, J. Domain Generalization via Heckman-type Selection Models. In The Eleventh International Conference on Learning Representations.

## 1. Installation
Please see requirements.txt, which presents the names and versions of all libraries that need to implement this repository. Please note that We mainly use the Pytorch backend libraries as follows:
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
We first need to download benchmark image (**WILDS**) data. Please run the following code. 

``` bash
# Run download_wilds_data.py

python download_wilds_data.py --root_dir ./data/benchmark/wilds
```
### WILDS benchmark
WILDS benchmark includes four datasets; Camelyon 17, PovertyMap, iWildCam, and RxRx1. Below are the details of each data.
- Camelyon17: Binary (tumor) classification.
- PovertyMap: Regression (wealth index prediction).
- iWildCam: multiclass (animal species) classification.
- RxRx1: multiclass (genetic treatments) classification.

![image](https://user-images.githubusercontent.com/36376255/226856940-2cca2f56-abee-46fa-9ec9-f187c6ac290b.png)

## 3.Experiments
please go to [code](2.run-cameloyon17-CNN-OneStep-HeckmanDG.py) and run it as follows:

```bash
# Run Heckman DG on Camelyon17 data with (batch_size, 3, 96, 96) input image and binary outcome

python 2.run-cameloyon17-CNN-OneStep-HeckmanDG.py
```

### Brief introduction of the code: This code is composed of the following 4 steps.

**1. Experiment Settings**
- Here, we set the name of data (camelyon17) and data-specific hyperparameters. The recommended data-specific hyperparameters are already set, so if you want to see results with other settings, please modify the **ars** variable in the code. 

**2. Data Preparation**
- The WILDS data basically require a large computing memory for the training step. If you want to test this code with the smaller size of data (subsets of the original data), please add (or uncomment) the following code at lines 50 to 54.

```python
# (2) run the experiment with a subset of data to test the implementation of HeckmanDG (take a small amount of memory)
if True:
    train_loader, valid_loader, test_loader = sub_dataloaders(train_loader, valid_loader, test_loader)
```

**3. HeckmanDG**
- Here, we initialize the network (CNN) and optimizer and run the Heckman DG model.

**4. Result Analysis**
- The results of this code are as follows:
  - plots of the training loss [learning curve](results/plots/HeckmanDG_camelyon17_loss.pdf)
  - plots of the probits [histogram](results/plots/HeckmanDG_camelyon17_probits.pdf)
  - AUC scores [prediction results](results/prediction/HeckmanDG_camelyon17.csv)

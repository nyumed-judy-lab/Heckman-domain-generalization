# image-benchmark-domain-generalization

This repository provides the PyTorch implementation of Heckman DG on WILDS benchmark data. We use the one-step optimization to train the Heckman DG model. In the Heckman DG model, the selelction (g) and outcome (f) networks are composed of data-specific convolutional neural networks (CNN) structures recommended by WILDS paper. In addition, we follow the hyperparaters of each data and model recommeded by ICLR paper (reference here) and the WILDS paper (reference here). Please note that the current repository provised onyl the code to run the experiment of Camelyon17 data. The codes for other data will be added. 

## References
Please refer to the following papers to set hyperparameters and reproduce experiments on the WILDS benchmark.
[GitHub Homepage]

- [WILDS](https://proceedings.mlr.press/v139/koh21a) paper: Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., ... & Liang, P. (2021, July). Wilds: A benchmark of in-the-wild distribution shifts. In International Conference on Machine Learning (pp. 5637-5664). PMLR.

- [HeckmanDG publihsed in ICLR 2023](https://openreview.net/forum?id=fk7RbGibe1) paper: Kahng, H., Do, H., & Zhong, J. Domain Generalization via Heckman-type Selection Models. In The Eleventh International Conference on Learning Representations.

## 1. Installation
Please see requirements.txt which presents the names and versions of all libraries that need to implement this repository. We mainly use the Pytorch backend libraries as follows:
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
# Run download_wilds_data.py as follows:

python download_wilds_data.py --root_dir ./data/benchmark/wilds
```
### WILDS bechmark
WILDS bechmark include 4 datasets; Camelyon 17, PovertyMap, iWildCam, and RxRx1. Below is the details of each data.
- Camelyon17: Binary (tumor) classification.
- PovertyMap: Regression (wealth index prediction)
- iWildCam: multiclass (animal species) classification.
- RxRx1: multiclass (genetic treatments) classification

![image](https://user-images.githubusercontent.com/36376255/226856940-2cca2f56-abee-46fa-9ec9-f187c6ac290b.png)

## 3.Experiments
```bash
# Run Heckman DG on Camelyon17 data with (batch_size, 3, 96, 96) input image and binary outcome

python 2.run-cameloyon17-CNN-OneStep-HeckmanDG.py
```

1. Data Input: write arguments (hyperparameters)
2. run Heckman DG
3. result analysis


## 4. Outputs
1. plots of training loss (learning curve).pdf and probits (histogram).pdf
3. prediction results.csv


# image-benchmark-domain-generalization

This repository is for the implementation of HeckmanDG on benchmark image data. We use the one-step optimization to train the Heckman DG model and the prediction network is composed of convolutional neural networks (CNN). We use the data-specific CNN structures recommended by our ICLR paper (reference here) and the WILDS paper (reference here). The current repo contains the code to run the experiment of Camelyon17 data.  

## Data Preparation
We first need to download benchmark image data. Run download_wilds_data.py as follows:

'''
python download_wilds_data.py --root_dir ./data/benchmark/wilds
'''

## Requirements
Please see requirements.txt which presents the names and versions of all libraries that need to implement this repository. We mainly use the Pytorch backend libraries as follows:
- torch==1.10.0
- torchaudio==0.10.0
- torchmetrics==0.11.1
- torchvision==0.11.0

## Experiments
1. Data Input: write arguments (hyperparameters)
2. run Heckman DG
3. result analysis

python 2.run-cameloyon17-CNN-OneStep-HeckmanDG


## Outputs
1. plots of training loss (learning curve).pdf and probits (histogram).pdf
3. prediction results.csv

# image-benchmark-domain-generalization

This repository provides the PyTorch implementation of Heckman DG on benchmark image data. We use the one-step optimization to train the Heckman DG model and the prediction network is composed of convolutional neural networks (CNN). We use the data-specific CNN structures recommended by our ICLR paper (reference here) and the WILDS paper (reference here). The current repo contains the code to run the experiment of Camelyon17 data.  

** 1. Installation ** 
```bash
# pip
pip install -r requirements.txt

# conda
conda install --file requirements.txt
```

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

```bash
# Run Heckman DG on Camelyon17 data with (batch_size, 3, 96, 96) input image and binary outcome

python 2.run-cameloyon17-CNN-OneStep-HeckmanDG.py
```


## Outputs
1. plots of training loss (learning curve).pdf and probits (histogram).pdf
3. prediction results.csv

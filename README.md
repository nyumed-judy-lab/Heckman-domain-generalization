# image-benchmark-domain-generalization

This repository is for the implementation of HeckmanDG on benchmark image data. We use the one-step optimization to train the Heckman DG model and the prediction network is composed of convolutional neural networks (CNN). We use the data-specific CNN structures recommended by our ICLR paper (reference here) and the WILDS paper (reference here). The current repo contains the code to run the experiment of Camelyon17 data.  

##### Data Preparation
We first need to download benchmark image data. 

1. run download_wilds_data.py

'''
python download_wilds_data.py --root_dir ./data/benchmark/wilds
'''

##### Python libraries
Please see Heckman.yaml (link here). We mainly use the Pytorch backend libraries.

##### main_heckmandg.py
1. Data Input: write arguments (hyperparameters)
2. run Heckman DG
3. result analysis

'''
python main-CNN-OneStep-Separated-HeckmanDG.py 
'''

##### output
1. trained model.
2. learning curve (plots)
3. experiment results.csv (prediction performances)

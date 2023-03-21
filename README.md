# image-benchmark-domain-generalization

This repository is for the implementation of HeckmanDG on benchmark image data. We use the one-step optimzation to train the Heckman DG model and the prediction newtwork is composed of convolotional nerual netowrks (CNN). We use the data-specific CNN structues that recommaned by our ICLR paper (reference here) and the WILDS paper (reference here). Current repo contains the code to run experiment of Cameylyon17 data.  

##### Data Preparation
We first need to download bechmark image data. 

1. run download_wilds_data.py

'''
python download_wilds_data.py --root_dir ./data/benchmark/wilds
'''

##### Python libraries
Please see heckmagdg.yaml (link here). We mainly use the pytorch backend libraries.

##### main_heckmandg.py
1. Data Input: write arguments (hyperparameters)
2. run heckman DG
3. result analysis

'''
python main-CNN-OneStep-Seperated-HeckmanDG.py 
'''

##### output
1. trained model.
2. learning curve (plots)
3. experiemnt results.csv (prediction performances)

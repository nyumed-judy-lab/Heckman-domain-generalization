# Heckman-domain-generalization
<!-- 
## References
Please refer to the following papers to set hyperparameters and reproduce experiments on the WILDS benchmark.
- [WILDS](https://proceedings.mlr.press/v139/koh21a) paper: Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., ... & Liang, P. (2021, July). Wilds: A benchmark of in-the-wild distribution shifts. In International Conference on Machine Learning (pp. 5637-5664). PMLR.
- [HeckmanDG](https://openreview.net/forum?id=fk7RbGibe1) paper: Kahng, H., Do, H., & Zhong, J. Domain Generalization via Heckman-type Selection Models. In The Eleventh International Conference on Learning Representations.

* ONE FILE for all datasets (and the readme.md file)
* What are the input and output of each step?
* What is the shape of the data (image, tabular)
* what functions for what
-->

This repository provides the PyTorch implementation of Heckman DG. We use the one-step optimization to train the Heckman DG model. In the Heckman DG model, the selection (g) and outcome (f) networks are composed of neural network structures. 
- For tabular datasets, we use the multi-layer NN. 
- For image datasets, we use data-specific convolutional neural networks (CNN) structures recommended by [WILDS](https://proceedings.mlr.press/v139/koh21a) paper. In addition, we follow the hyperparameters of each data and model recommended by [HeckmanDG](https://openreview.net/forum?id=fk7RbGibe1) paper.  

## Installation
Please see [requirements.txt](requirements.txt). It presents the names and versions of all libraries that we have to install before the implementation of this repository. Please note that we mainly use the Pytorch backend libraries as follows:
- torch==1.10.0
- torchaudio==0.10.0
- torchmetrics==0.11.1
- torchvision==0.11.0

```bash
# pip
pip install -r requirements.txt
```

## **Data Preparation**
Please put your data in [data](data). If you want to apply **structured (tabular)** data, please put your data in [data](data). If you want to use **WILDS** benchmark, please run the following code to download it on [wilds](data/benchmark/wilds). 

``` bash
# Run download_wilds_data.py
python download_wilds_data.py --root_dir ./data/benchmark/wilds
```

## **Experiments**
Please go to [main_heckmandg.py](main_heckmandg.py). The experiment is composed of the following 4 steps; (1) Experiment Settings, (2) Data Preprocessing, (3) Heckman DG, (4) Result Analysis.


### **1. Experiment Settings**
- Here, we set the **data_name** (e.g. insight or camelyon17), **data_shape** (tabular or image), and hyperparameters. You can set hyperparameters with arguments named **args** consisting of the learning rate, weight decay, and optimizer. Please note that recommended data-specific hyperparameters are already set for the INSIGHT and WILDS benchmark, so if you want to see results with other settings, please modify the **args** variable in the [argmarser.py](utils/argparser.py). 

- The <font color="blue">**input**</font> of data_argument function is the data_name and the <font color="blue">**output**</font> of the data-specific arguments are configuration sets and the data_type ('tabular' or 'image'). 

 ```python
# data-specific arguments 
args, data_type = data_argument(data_name)
```

### **2. Data Preprocessing**
This repository provides HeckmanDG for two data types, including (1) tabular, and (2) image data.

#### **2.1 Preprocessing of Tabular data**
**Tabular data** is a specific form of structured data that is organized into rows and columns. Each row represents an observation, and each column represents a variable. The shape of the tabular data is (# observations, # variables), and below is an example of tabular data with the shape of (# observation: 3, # variables: 3).

```bash
[[10, 10, 10],
[10, 10, 10],
[10, 10, 10],]
```

This repository provides the functions that can perform the **standardization** and the **Missing value imputation**. 

- **Standardization**: transforms the input data into a mean of zero and a standard deviation of one. To apply standardization to the training, validation, and testing data, you would need to follow these steps:
  1. Calculate each feature's mean and standard deviation (column) in the training data.
  2. Transform the training data by subtracting the mean and dividing it by the standard deviation for each feature. This will center the data around zero and scale it to have a standard deviation of one.
  3. Use the same mean and standard deviation values to transform the validation and testing data. This is important to ensure that the validation and testing data are processed in the same way as the training data.


```python
# In this repository, we use the function StandardScaler
from sklearn.preprocessing import StandardScaler
# Create a StandardScaler object
scaler = StandardScaler()
# Fit the scaler to the training data and transform it
scaler_fitted = scaler.fit(x_train)
x_train_scaled = scaler_fitted.transform(x_train)
# Use the same scaler to transform the validation and testing data
x_val_scaled = scaler_fitted.transform(x_val)
x_test_scaled = scaler_fitted.transform(x_test)
```

- **Missing value imputation**
We use the **SimpleImputer(strategy='mean')** for the missing value imputation (you can change the strategy 'median', 'most_frequent', 'constant'). 
  - If “mean”, then replace missing values using the mean along each column. It can only be used with numeric data.
  - If “median”, then replace missing values using the median along each column. It can only be used with numeric data.
  - If “most_frequent”, then replace missing using the most frequent value along each column. It can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.

#### **2.2 Preprocessing of Image (WILDS benchmark) data**#### 

**Image data**: Image data is structured in a 3D format. The **3D format** refers to an image that has three dimensions (# observations, # channels, width, height), and each image has the shape of the (#channels, width, height). We use Pytorch **data_loader** that can put a subset (minibatch) of data to the model in the training process, so the data shape would be (# batch_size, # channels, width, height). Below is the example of an image having the shape of the (#channels: 3, width: 3, height: 3).


```bash
red
[[10, 10, 10],
[10, 10, 10],
[10, 10, 10],]
green
[[10, 10, 10],
[10, 10, 10],
[10, 10, 10],]
blue
[[10, 10, 10],
[10, 10, 10],
[10, 10, 10],]
```
The fuction **IdentityTransform** provides data-specific nomalization and augmentation modules. The input and output of the function is the raw image dataset and the normalized and augmented dataset.
Please see the details in [tranforms](utils_datasets/transforms.py).

[tranforms](utils_datasets/transforms.py) includeds **Data Normalization** module having pre-defined mean and std values for each domain and channel, and **Data Augmentation** module for each dataset as follows:


<img width="837" alt="hyperparameters" src="https://user-images.githubusercontent.com/36376255/229375372-3b3bd721-b5f2-405a-9f5e-02966dc20cd6.png">

This figure represents hyperparameters of the two-step optimization of the Heckman DG. Cells with two entries denote that we used different values for training domain selection and outcome models. In this repository, for the one-step optimization, we followed the hyperparameters of the outcome model.
![image](https://user-images.githubusercontent.com/36376255/226856940-2cca2f56-abee-46fa-9ec9-f187c6ac290b.png)





### **3. HeckmanDG**
- Here, we initialize the neural networks (NNs) and run the Heckman DG model. We use deep neural networks (DNNs; a.k.a. multi-layer perceptron) and convolutional neural networks (CNNs) for the tabular data and image data, respectively.

##### Tabular Data: **HeckmanDNN** 
For the tabular data, you need to import the **HeckmanDNN** function. The **HeckmanDNN** contains the selection model (g_layers) and outcome model (f_layers), so please put the number of nodes and layers to construct them. This is an example of the **HeckmanDNN**.

```python
network = HeckmanDNN([tr_x.shape[1], 128, 64, 32, 1], 
                     [tr_x.shape[1], 64, 32, 16, args.num_domains], 
                     dropout=0.5, 
                     batchnorm=True, 
                     activation='ReLU')
```


##### Image Data: **HeckmanCNN**. 
For the image data, you need to import the **HeckmanCNN**. The function of **HeckmanCNN** contains various CNN structures. The input of this **HeckmanCNN** is the argument named **args**. 

```python
network = HeckmanCNN(args)
```

Both networks are put into the **HeckmanBinaryClassifier**, and the output is the model (object), and it is saved in the results folder.

### **4. Result Analysis**
- The results of this code are as follows:
  - plots of the training loss [learning curve](results/plots/HeckmanDG_camelyon17_loss.pdf)
  - plots of the probits [histogram](results/plots/HeckmanDG_camelyon17_probits.pdf)
  - AUC scores [prediction results](results/prediction/HeckmanDG_camelyon17.csv)

- From the function of **plots_loss**, we can see the following [learning curve](results/plots/HeckmanDG_camelyon17_loss.pdf). The x-axis represents the epoch, and the y-axes are loss, auroc, and rho trajectory in the training process.

![image](https://user-images.githubusercontent.com/36376255/229378713-62511fcb-be4a-4973-a6e5-21029765d3fa.png)

- From the function of **plots_probit**, we can see the [histogram](results/plots/HeckmanDG_camelyon17_probits.pdf) that represents the distribution of probits for each domain. Please refer to the function of **model.get_selection_probit** yields the probits.

![image](https://user-images.githubusercontent.com/36376255/229378704-24477849-d9ce-49c7-bf0a-97724fcd7c81.png)

<!-- 
 **IMAGE**: The WILDS data basically require a large computing memory for the training step. If you want to test this code with the smaller size of data (subsets of the original data), please add (or uncomment) the following code at lines 50 to 54.

The **args** contains the name of data, backbone, and hyperparameters (learning rate, etc.). For the WILDS data, the data-specific arguments are already set.



**Three prediction tasks**: (1) binary classification (Camelyon17), (2) multicalss classification(iWildCam, RxRx1), and regression (PovertyMap), on WILDS benchmark data as follows:
- Camelyon17: Binary (tumor) classification.
- iWildCam: multiclass (animal species) classification. (In preparation)
- RxRx1: multiclass (genetic treatments) classification. (In preparation)
- PovertyMap: Regression (wealth index prediction). (In preparation)

In addition, this repository provides the data-specific normalization and augmentation functions as follows:
- Camelyon17: N/A
- PovertyMap: Color jittering
- iWildCam: RandAugment
- RxRx1: RandAugment

### WILDS benchmark

Summary of the four datasets from the WILDS benchmark. 
![image](https://user-images.githubusercontent.com/36376255/226856940-2cca2f56-abee-46fa-9ec9-f187c6ac290b.png)

<img width="837" alt="hyperparameters" src="https://user-images.githubusercontent.com/36376255/229375372-3b3bd721-b5f2-405a-9f5e-02966dc20cd6.png">

This figure represents hyperparameters of the two-step optimization of the Heckman DG. Cells with two entries denote that we used different values for training domain selection and outcome models. In this repository, for the one-step optimization, we followed the hyperparameters of the outcome model.

#### With your own data

- To input your own image data, you need to customize the preprocessing code in the 
-->

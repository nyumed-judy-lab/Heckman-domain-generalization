# **Heckman-domain-generalization**
This repository provides the PyTorch implementation of Heckman DG. Need to first install required libraries and download (or put) datasets in [data](./data/).

## **Installation**
Before running the code in this repository, you will need to install the required dependencies listed in [requirements.txt](requirements.txt). This file contains the names and versions of all libraries that we have to install. We mainly use PyTorch backend libraries, and the following versions are recommended:
- torch==1.10.0
- torchaudio==0.10.0
- torchmetrics==0.11.1
- torchvision==0.11.0

To install these dependencies, simply run the following command:

```bash
pip install -r ./requirements.txt 
conda install -f ./requirements.txt
```

## **Data Preparation**
To prepare your data for use with this repository, please place your data in the [data](data) directory. 
- If you plan to use **structured (tabular)**, please place your data in the [data](data) directory.
- If you plan to use the WILDS benchmark data, please follow these steps to download the data:
 1. Create a directory named wilds inside the [wilds](data/benchmark/wilds) directory.
 2. Run the following command to download the data:

``` bash
# Run download_wilds_data.py
python download_wilds_data.py --root_dir ./yourdirectory/
```

## **Heckman DG**
Please go to [main_heckmandg.py](main_heckmandg.py) for the implementation of the HeckmanDG on tabular or image data. The code performs the following four steps:
 1. Experiment Settings
 2. Data Preprocessing
 3. Heckman DG
 4. Evaluation


### **1. Experiment Settings**
This section imports the necessary **Modules** and data-specific **Arguments**. 

#### **1.1 Modules**
- [x] 1. Data Preparation & Preprocessing
  - [x]  1. Tabular: INSIGHT
  - [x]  2. Image: Cameyloyon17, Povertypmap, iWildCam, Rxrx1
- [x] 2. Data-specific Nerual Networks
  - [x]  1. Heckman_DNN
  - [x]  2. Heckman_CNN
- [ ] 3. Heckman DG Training modules: 
  - [x]  1. HeckmanDG_DNN_BinaryClassifier (tabular)
  - [x]  2. HeckmanDG_CNN_BinaryClassifier (image)
  - [x]  3. HeckmanDG_CNN_Regressor (image)
  - [ ]  4. HeckmanDG_CNN_MultiClassifier: In preparation (image)
- [x]  4. Evaluation
  - [x]  Classification: Accuracy, F1 score, AUROC scores of Training, Validation, and Testing data
  - [x]  Regression: MSE, MAE, Pearsonr scores of Training, Validation, and Testing data
  - [x]  Plots: plot_probits, plot_loss

#### **1.2 Auguments**
- The ```data_name``` variable has to be set first among ```data_list=['insight', 'camelyon17', 'poverty', 'rxrx1', or 'iwildcam']``` . The code then calls the ```data_argument``` function to obtain the arguments for the selected dataset e.g., ```args =  data_argument(data_name)``` .

- The **input** of ```data_argument``` function is the ```data_name``` and the **output** is the data-specific arguments ```args``` having configurations and hyperparameters sets. 

- The ```args``` contains data-specific **Configuration** (train_domains, validation_domains, testing domains, num_classes, etc.) and **Hyperparameters** (e.g. data_name, data_type = (image or tabular), backbone = (densnet or etc.), batch_size, epochs, optimizer, learning_rate, weight_decay, augmentation, randaugment, etc.).

- Note that recommended data-specific hyperparameters are already set for the INSIGHT and WILDS benchmark in the functions (```args_insight```, ```args_cameloyn17```, ```args_poverty```, ```args_rxrx1```, ```args_iwildcam```), so if you want to see results with other settings, please create or modify the **args** variable in the [argmarser.py](utils/argparser.py). 

- The ```fix_random_seed(args.seed)``` function then sets the random seed to a fixed value in an implementation.

### **2. Data Preprocessing**
This repository provides HeckmanDG for two data types, including (1) tabular, and (2) image data.

#### **2.1 Preprocessing of Tabular data**
**Tabular data** is a specific form of structured data that is organized into rows and columns. Each row represents an observation, and each column represents a variable. The shape of the tabular data is (# observations, # variables), and below is an example of tabular data with the shape of (# observation: 3, # variables: 3).

```bash
[[10, 10, 10],
[10, 10, 10],
[10, 10, 10],]
```

For the tabular data, the function ```dataset = DatasetImporter(args)``` reads the ```dataset.feather``` file and then the imported dataset is separated the numerical and categorical columns. 

The list of ```train_domains``` has to be set manually e,g, ```['domain1', 'domain2', 'domain3', 'domain4']``` then the code sets the number of domains and split the data into training/validation/testing data stratiried by domain memberships.

****************************************
DOMAIN GENERALIZATION
HOW does User specify
HOW does Repo specify

+ add detailed exap
+ 10 domains
+ domain generalization
```train_domains``` = ```[domain1, domain 2, or 3]```
```valid_domains``` = ```[domain1, domain 9]```
```test_domains``` = ```[domain 10]```
# training domians: 1, 2, 3 -> training_data / validation_data (hold_out)
# testing domians: 4
# int validation: socre on testing of domain 1, 2, 3
# ext nvalidation: socre on testing of domain 4
What dataset need to be uploaded

****************************************



The ```precessed_data=preprocessing_tabular(datset)``` function then split the data into training and validation sets and applies **scaling** and **imputation** to the numerical (continuous) variables. 

**1. Standardization**: ```scaler = StandardScaler()``` transforms the input data into a mean of zero and a standard deviation of one. To apply standardization to the training, validation, and testing data, you would need to follow these steps:
  1. ```scaler.fit(x_train)```: Calculate each feature's mean and standard deviation (column) in the training data.
  2. ```scaler.transform(x_train)``` Transform the training data by subtracting the mean and dividing it by the standard deviation for each feature. This will center the data around zero and scale it to have a standard deviation of one.
  3. ```scaler.transform(x_valid or x_test)```: Use the same mean and standard deviation values to transform the validation and testing data. This is important to ensure that the validation and testing data are processed in the same way as the training data.

**2. Missing value imputation**: ```imputer = SimpleImputer(strategy='mean')``` performs the missing value imputation (you can change the strategy 'median', 'most_frequent', 'constant'). 
  - If “mean”, then replace missing values using the mean along each column. It can only be used with numeric data.
  - If “median”, then replace missing values using the median along each column. It can only be used with numeric data.
  - If “most_frequent”, then replace missing using the most frequent value along each column. It can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.

#### **2.2 Preprocessing of Image (WILDS benchmark) data**
**Image data**: is structured in a 3D format. The **3D format** refers to an image that has three dimensions (# channels, width, height) and image datasets has four dimensions (# observations, # channels, width, height). We use Pytorch **data_loader** that can put a subset (minibatch) of data to the model in the training process, so the data shape would be (# batch_size, # channels, width, height). Below is the example of an image having the shape of the (#channels: 3, width: 3, height: 3).

```bash
[
# red
[[10, 10, 10],
[10, 10, 10],
[10, 10, 10],]
# green
[[10, 10, 10],
[10, 10, 10],
[10, 10, 10],]
# blue
[[10, 10, 10],
[10, 10, 10],
[10, 10, 10],]
]
```

For the image data, the function ```dataset = DatasetImporter(args)``` reads the data and it has data-speficic modules:
 - ```dataset = WildsCamelyonDataset()```
 - ```dataset = WildsPovertyMapDataset()```
 - ```dataset = WildsRxRx1Dataset()```
 - ```dataset = WildsIWildCamDataset()```

The output ```datsets``` is put into the ```dataloaders(args, dataset)``` function and it then return  ```train_loader, valid_loader, test_loader``` to train the model with mini-batch data (subsets of data).
 
<!-- <img width="837" alt="hyperparameters" src="https://user-images.githubusercontent.com/36376255/229375372-3b3bd721-b5f2-405a-9f5e-02966dc20cd6.png">

This figure represents hyperparameters of the two-step optimization of the Heckman DG. Cells with two entries denote that we used different values for training domain selection and outcome models. In this repository, for the one-step optimization, we followed the hyperparameters of the outcome model.

![image](https://user-images.githubusercontent.com/36376255/226856940-2cca2f56-abee-46fa-9ec9-f187c6ac290b.png)
-->
### **3. HeckmanDG**
The code imports two neural network architectures: ```HeckmanDNN``` and ```HeckmanCNN```. These architectures are used to create the network used by the models. The code also imports several models (```HeckmanDG_DNN_BinaryClassifier```, ```HeckmanDG_CNN_BinaryClassifier```, ```HeckmanDG_CNN_Regressor```, ```HeckmanDG_CNN_MultiClassifier```) which utilize the Heckman correction for domain generalization.

We first initialize the neural networks (NNs) and then run the Heckman DG model. We use deep neural networks (DNNs; a.k.a. multi-layer perceptron) and convolutional neural networks (CNNs) for the tabular data and image data, respectively. We use the **one-step optimization** to train the Heckman DG model. The Heckman DG model has selection (g) network to predict domains and the outcome (f) network to predict label (class, multi-class, or countinous variable for binary classification, multiclass classification, and regression, respectively). This repository provides data-specific convolutional neural networks (CNN) structures recommended by [WILDS](https://proceedings.mlr.press/v139/koh21a) paper. In addition, we follow the hyperparameters of each data and model recommended by [HeckmanDG](https://openreview.net/forum?id=fk7RbGibe1) paper.  

#### **3.1 Tabular Data**
For the tabular data, the code (1) creates a ```HeckmanDNN``` network with the specified layers, and (2) trains a ```HeckmanDG_DNN_BinaryClassifier``` model on the data. The model is trained using the fit function with the specified training and validation data.

##### **3.1.1 Initiailize the Network**: 
The ```network = HeckmanDNN(args)``` is defined. The **HeckmanDNN** contains the selection model (g_layers) and outcome model (f_layers). Please put the number of nodes and layers to construct them like ```HeckmanDNN(f_network_structure, f_network_structure)```.  The initialized network is put into the ```HeckmanDG_DNN_BinaryClassifier(network)```

  - ```f_network_structure```: The first and last layers has to be composed of the number of neurons equal to the number of columns and the number of training domains in data (e,g, ```[tr_x.shape[1], 64, 32, 16, args.num_domains]```.)
  - ```g_network_structure```: The first and last layers has to be composed of the number of neurons equal to the number of columns and the number of classes in data ```[tr_x.shape[1], 64, 32, 16, args.num_classes]```

<!-- The ```HeckmanDNN``` module has the following Parameters & Attrbutes:
  - ```f_layers```(PyTorch Sequential module): a list of integers that define the architecture of the DNN for the outcome model.
  - ```g_layers```(PyTorch Sequential module): a list of integers that define the architecture of the DNN for the selection model.
  - ```rho```(PyTorch Parameter):  representing the selection equation coefficient.
  - ```activation```: the activation function to use in the hidden layers (default: nn.ReLU).
  - ```dropout```: the dropout probability to use in the hidden layers (default: 0.0).
  - ```batchnorm```: whether to use batch normalization in the hidden layers (default: False).
  - ```bias```: whether to include bias in the linear layers (default: True).
  - **functions**
   - ```forward```: takes an input tensor x and returns the concatenation of the outputs of the DNNs for the outcome model (f) and the selection model (g).
   - ```forward_f```: takes an input tensor x and returns the output of the DNN for the outcome model.
   - ```forward_g```: takes an input tensor x and returns the output of the DNN for the selection model. 
-->

##### **3.1.2. Train the model** 
The ```HeckmanDG_DNN_BinaryClassifier(network, optimizer, and scheduler)``` model is then defined with the network, optimizer, and scheduler as input, and the model is trained on the training data using the ```fit``` function.  The ```HeckmanDG_DNN_BinaryClassifier``` class takes four parameters as input: (1) ```network```, (2) ```optimizer```, (3) ```scheduler```, and (4) ```config```.
 - ```network```: the deep neural network that is used to perform the classification task
 - ```optimizer```: the optimizer used for backpropagation and gradient descent during training
 - ```scheduler```: the learning rate scheduler for the optimizer
 - ```config```: a dictionary containing additional configuration parameters like device (cpu or gpu), maximum number of epochs, and batch size.

The ```fit``` function trains the classifier on the given data, which is a ```dictionary``` containing the training and validation data. It then initializes the optimizer, learning rate, scheduler, and sets up empty lists to store the training and validation loss and AUC scores. Below is the traning process:

 0. the ```fit``` first creates a data loader for each of the train and validation datasets using the ```DataLoader``` class from the ```torch.utils.data module```. The model is trained using the given ```train_loader``` and ```valid_loader``` . The training is perfomed using mini-batches of data. 
 1. For each mini-batch of data, the model is trained using loss function. After the training of the model, the method calculates the loss and AUC score for the validation data using the trained model. The loss and scores (auc score) for both training and validation data are stored in the ```train_loss_traj```, ```valid_loss_traj```, ```train_score_traj```, and ```valid_score_traj``` lists. The code also stores the model's parameters that produce the lowest validation loss and the highest validation accuracy. 
 2. Checks if the current validation loss is lower than the best validation loss seen so far. If so, it saves the current state of the network as the best model. 
 3. The ```fit()``` returns the best model. The final trained model object is saved in [results](./results/).

### **3.2 Image Data**
For the image data, the code (1) creates a ```HeckmanCNN``` network with the specified arguments, and (2) trains either a ```HeckmanDG_CNN_BinaryClassifier```, ```HeckmanDG_CNN_Regressor```, or ```HeckmanDG_CNN_MultiClassifier``` model depending on the ```data_name``` and ```loss_type```. The model is trained using the ```fit``` function with the specified training and validation data loaders.

#### **3.2.1 Initiailize the Network**
The  ```network = HeckmanCNN(args)``` is defined. The ```HeckmanCNN(args)``` also contains the selection model (g_layers) and outcome model (f_layers), and the data-specific CNN structures are already set. 

#### **3.2.2. Train the model** 
The initialized network is put into the following modules, and it is then defined with the network, optimizer, and scheduler as input
 - ```model = HeckmanDG_CNN_BinaryClassifier(args, network, optimizer, scheduler)```: for the binary classification (camelyon17),
 - ```model = HeckmanDG_CNN_Regressor(args, network, optimizer, scheduler)```: for the regression (povertymap),
 - ```model = HeckmanDG_CNN_MultiClassifier(args, network, optimizer, scheduler)```: for the multinomial classification,(iWildCam, RxRx1)

The model is trained on the training data using the ```fit``` function performing the following steps: 
 0. The ```fit()``` function is used to train the model using the given ```train_loader``` and ```valid_loader``` . The training is done using mini-batches of data. The ```fit()``` first initializes the ```optimizer``` and ```scheduler``` for the training process. Then it loops through the epochs of training, and for each epoch, it loops through the mini-batches of data in the training data loader.
 1. For each mini-batch of data, the function **InputTransforms** first provides data-specific normalization and augmentation modules. The input and output of the function is the raw image dataset and the normalized and augmented dataset. Please see the details in [tranforms](utils_datasets/transforms.py) that include ```Data Normalization``` module having pre-defined mean and std values for each domain and channel, and ```Data Augmentation``` module for each dataset as follows (as the hyperparameters, we can select whether we are going to perform augmentation or not with the bool type variable of ```args.augmentation``` (but note that it is all already set):
  - ```Camelyon17Transform```: Transforms for the Camelyon17 dataset. (normalization)
  - ```PovertyMapTransform()```: Transforms (Color jittering) for the Poverty Map dataset.
  - ```RxRx1Transform```: Transforms (RandAugment) for the RxRx1 dataset.
  - ```IWildCamTransform```Transforms (RandAugment) for the iWildCam dataset.
 2. For each transformed mini-batch of data, the model is trained using the loss function of each task (classification, regression, and multiclass classification). After the training of the model, the method calculates the loss and AUC score for the validation data using the trained model. The loss and scores (F1 score and accuracy for the classifier, and Pearson for the regressor) for both training and validation data are stored in the ```train_loss_traj```, ```valid_loss_traj```, ```train_score_traj```, and ```valid_score_traj``` lists. The code also stores the model's parameters that produce the lowest validation loss and the highest validation accuracy. 
 3. Checks if the current validation loss is lower than the best validation loss seen so far. If so, it saves the current state of the network as the best model. 
 4. The ```fit()``` returns the best model. The final trained model object is saved in [results](./results/).

### **4. Evaluation**
This section evaluates the trained model on the training, validation, and test data. 

- For tabular data, the code calculates the AUC score for each domain in the test dataset and records the scores for all domains. The DataFrame is then saved to a CSV file in the ./results/prediction/ directory. The mean score for internal, external, and all sites are also calculated and included in the DataFrame.
+++++ save prediction results (output of the model)

- For image data, the ```prediction()``` function calculates the AUC score, F1 score, and accuracy for the train, validation, and test sets. The scores are recorded in a pandas DataFrame and saved to a CSV file in the [results](./results/prediction/) directory.

- ```plots_loss()``` generates a plot of the training loss for each domain in a different color, and 

- ```plots_probit()``` generates a plot (histogram) of the distribution of the selection probits for each domain. This function uses ```model.get_selection_probit()``` function that can yield the probits.

The results and plots are saved to the [results](./results/) directory, with the filenames containing the name of the algorithm used (HeckmanDG) and the type of data (```args.data```) and here are the examples:
  - plots of the training loss [learning curve](results/plots/HeckmanDG_camelyon17_loss.pdf)
  - plots of the probits [histogram](results/plots/HeckmanDG_camelyon17_probits.pdf)
  - AUC scores [prediction results](results/prediction/HeckmanDG_camelyon17.csv)

<!-- 
![image](https://user-images.githubusercontent.com/36376255/229378704-24477849-d9ce-49c7-bf0a-97724fcd7c81.png) 

![image](https://user-images.githubusercontent.com/36376255/226856940-2cca2f56-abee-46fa-9ec9-f187c6ac290b.png)

<img width="837" alt="hyperparameters" src="https://user-images.githubusercontent.com/36376255/229375372-3b3bd721-b5f2-405a-9f5e-02966dc20cd6.png">

This figure represents hyperparameters of the two-step optimization of the Heckman DG. Cells with two entries denote that we used different values for training domain selection and outcome models. In this repository, for the one-step optimization, we followed the hyperparameters of the outcome model.
-->


<!-- NOTES
**IMAGE**: The WILDS data basically require a large computing memory for the training step. If you want to test this code with the smaller size of data (subsets of the original data), please add (or uncomment) the following code at lines 50 to 54.

**Three prediction tasks**: (1) binary classification (Camelyon17), (2) multicalss classification(iWildCam, RxRx1), and regression (PovertyMap), on WILDS benchmark data as follows:
- Camelyon17: Binary (tumor) classification.
- iWildCam: multiclass (animal species) classification. (In preparation)
- RxRx1: multiclass (genetic treatments) classification. (In preparation)
- PovertyMap: Regression (wealth index prediction). (In preparation)

## References
Please refer to the following papers to set hyperparameters and reproduce experiments on the WILDS benchmark.
- [WILDS](https://proceedings.mlr.press/v139/koh21a) paper: Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., ... & Liang, P. (2021, July). Wilds: A benchmark of in-the-wild distribution shifts. In International Conference on Machine Learning (pp. 5637-5664). PMLR.
- [HeckmanDG](https://openreview.net/forum?id=fk7RbGibe1) paper: Kahng, H., Do, H., & Zhong, J. Domain Generalization via Heckman-type Selection Models. In The Eleventh International Conference on Learning Representations.

* ONE FILE for all datasets (and the readme.md file)
* What are the input and output of each step?
* What is the shape of the data (image, tabular)
* what functions for what
-->

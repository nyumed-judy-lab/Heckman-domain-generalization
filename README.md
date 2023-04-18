# **Heckman-domain-generalization**
This repository provides the PyTorch implementation of Heckman DG. Need to first install required libraries and download (or put) datasets in [data](./data/).

## **Installation**
Before running the code in this repository, you will need to install the required dependencies listed in [requirements.txt](requirements.txt). This file contains the names and versions of all libraries that we have to install. We mainly use PyTorch backend libraries, and the following versions are recommended:

To install these dependencies, simply run the following command:

```bash
pip install -r ./requirements.txt

OR

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
conda install anaconda
pip install ray
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
  - [x]  4. HeckmanDG_CNN_MultiClassifier (image) (+ model selection framework)
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

- For the tabular data, the function ```dataset = DatasetImporter(args)``` reads the ```dataset.feather``` file and then the imported dataset is separated the numerical and categorical columns. 

- For the DOMAIN-GENERALIZATION EXPERIEMNT SETTING, please creates a list of domains to use for training the model e.g. ```train_domains = ['domain 1', ..., 'domain K']```. For the INSIGHT data, the list of ```train_domains``` is already set as ```['A05', 'A07', 'B03', 'C05']```. The next codes ```args.train_domains = train_domains``` and ```args.num_domains = len(train_domains)``` set the values of the ```train_domains``` and ```num_domains``` variable in the ```args``` namespace.

- For the tabular data, this repository performs in-distribution (ID) validation for the model selection in the training process and internal/external validations for the model evaluation in the tesing process. Hence, the validaation domains are equal to training domains and the testing domains are the rest domains differ from the training domains (for the external validation). Below is an example of domain sets if we have 5 differnt domains in data.
 - ```train_domains``` = ```[domain1, domain 2, domain 3, domain 4]``` (user specifies) 80 100%
 - ```valid_domains``` = ```[domain1, domain 2, domain 3, domain 4]``` (repo specifies) 20 100%
 - ```test_domains``` = ``` domain 5 (external validation)]``` (repo specifies) 100%

Then, the code split the data into training/validation/testing data stratiried by domain memberships.

- ```train_val_dat, test_dat = train_test_split(dataset, stratify=dataset['SITE']+dataset['CVD'].astype(str), test_size=args.test_size, random_state=args.seed)```: Splits the dataset into training/validation and testing dataset using the ```train_test_split``` function. The ```stratify``` parameter ensures that the proportions (```args.test_size = 0.2```) of each domain and target (e.g. CVD) label are roughly equal in the training/validation and testing dataset.

- ```tr_x, tr_s, tr_y, val_x, val_s, val_y, num_imputer, scaler = preprocessing_tabular(train_val_dat, args, num_cols, cat_cols)```: ```preprocessing_tabular``` has also the function ```train_test_split``` that splits ```train_val_dat``` into ```train_dat, val_dat```. The training and validation sets are then put into the modules of **scaling** and **imputation**. The function finally returns preprocessed data (tr_x, tr_s, tr_y, val_x, val_s, val_y) as well as a fitted numerical ```imputer``` and ```scaler``` for use in preprocessing the test set. Below is  how the preprocessing is performed:

  A B C D E INT EXT ALL DIFF
A - - - - -
B - - - - -
C - - - - -
D - - - - -
E - - - - -

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

- For the image data, the function ```dataset = DatasetImporter(args)``` reads the data (```torch```) using data-speficic modules:
 - Camelyon17: ```dataset = WildsCamelyonDataset()```-> Binary (tumor) classification.
 - PovertyMap: ```dataset = WildsPovertyMapDataset()``` -> Regression (wealth index prediction).
 - RxRx1: ```dataset = WildsRxRx1Dataset()``` -> multiclass (genetic treatments) classification.
 - iWildCam: ```dataset = WildsIWildCamDataset()``` -> multiclass (animal species) classification. 

- DOMAIN-GENERALIZATION EXPERIMENT SETTING: In the case of the image data, this repository performs external validation (doesn't perform internal validation). That is. testing data is the data of different domains from training domains) and the sets of domains are already set. The data-import modules of ```Camelyon``` and ```Povertymap``` data requires specific training/validation/testing domains, e.g. ```dataset = WildsCamelyonDataset(root=args.root, train_domains=args.train_domains, validation_domains=args.validation_domains, test_domains=args.test_domains)```, ```dataset = WildsPovertyMapDataset(root=args.root, train_domains = get_countires(args.fold)[0], validation_domains = get_countires(args.fold)[1], test_domains = get_countires(args.fold)[2])```. The ```Povertymap``` data is composed of five datasets; ```A: _SURVEY_NAMES_2009_17A, B: _SURVEY_NAMES_2009_17B, C: _SURVEY_NAMES_2009_17C , D: _SURVEY_NAMES_2009_17D, _SURVEY_NAMES_2009_17E```, so please set ```args.fold``` among the list of folds ```'A', 'B', 'C', 'D', 'E'```. The domain set of each dataset is determined by country information (please see the module ```SinglePovertyMap```. For the ```RxRx1``` and ```iWildcam``` data, the training/validation/testing domains are set in each module as lists of string values (```_train_domains_str```, ```_validation_domains_str```, ```_test_domains_str```) and indices of domains (```train_mask, id_val_mask, ood_val_mask, test_mask(```) Please see the modules ```WildsRxRx1Dataset``` and ```WildsIWildCamDataset```. If you want to try another domain set, please modify ```train_domains, validation_domains, test_domain``` in the module ```Wilds{dataname}Dataset```. 

Please note that for the image data, this repository performs the in-distribution (ID) validation for the selection model (g) selection and the out-of-distribution (OOD) validation for the outcome model (f) selection in the training process.

The output ```dataset``` is put into the ```dataloaders(args, dataset)``` function, and it then returns ```train_loader, valid_loader, test_loader``` to train the model with mini-batch data (subsets of data). 

<!-- <img width="837" alt="hyperparameters" src="https://user-images.githubusercontent.com/36376255/229375372-3b3bd721-b5f2-405a-9f5e-02966dc20cd6.png">

This figure represents hyperparameters of the two-step optimization of the Heckman DG. Cells with two entries denote that we used different values for training domain selection and outcome models. In this repository, for the one-step optimization, we followed the hyperparameters of the outcome model.

![image](https://user-images.githubusercontent.com/36376255/226856940-2cca2f56-abee-46fa-9ec9-f187c6ac290b.png)
-->

### **3. HeckmanDG**
The code imports two neural network architectures: ```HeckmanDNN``` and ```HeckmanCNN```. These architectures are used to create the network used by the models. The code also imports several models (```HeckmanDG_DNN_BinaryClassifier```, ```HeckmanDG_CNN_BinaryClassifier```, ```HeckmanDG_CNN_Regressor```, ```HeckmanDG_CNN_MultiClassifier```) which utilize the Heckman correction for domain generalization.

We first initialize the neural networks (NNs) and then run the Heckman DG model. We use deep neural networks (DNNs; a.k.a. multi-layer perceptron) and convolutional neural networks (CNNs) for the tabular data and image data, respectively. We use the **one-step optimization** to train the Heckman DG model. The Heckman DG model has selection (g) network to predict domains and the outcome (f) network to predict label (class, multi-class, or countinous variable for binary classification, multiclass classification, and regression, respectively). This repository provides data-specific convolutional neural networks (CNN) structures recommended by [WILDS](https://proceedings.mlr.press/v139/koh21a) paper. In addition, we follow the hyperparameters of each data and model recommended by [HeckmanDG](https://openreview.net/forum?id=fk7RbGibe1) paper.  

#### **3.1 Tabular Data**
For the tabular data, the code (1) creates a ```HeckmanDNN``` network with the specified layers, and (2) trains a ```HeckmanDG_DNN_BinaryClassifier``` model on the data. The model is trained using the ```fit``` function with the specified training and validation data.

#### **3.1.1 Initiailize the Network**: 
The ```network = HeckmanDNN(args)``` is defined. The **HeckmanDNN** contains the selection model (g_layers) and outcome model (f_layers). Please put the number of nodes and layers to construct them like ```HeckmanDNN(f_network_structure, f_network_structure)```.  The initialized network is put into the ```HeckmanDG_DNN_BinaryClassifier(network)```.

  - ```f_network_structure```: The first and last layers has to be composed of the number of neurons equal to the number of columns and the number of training domains in data (e,g, ```[tr_x.shape[1], 64, 32, 16, args.num_domains]```.)
  - ```g_network_structure```: The first and last layers has to be composed of the number of neurons equal to the number of columns and the number of classes in data ```[tr_x.shape[1], 64, 32, 16, args.num_classes]```. 
  - The ```HeckmanDNN``` module has the following Parameters & Attrbutes:
   - **functions**
    - ```f_layers```(PyTorch Sequential module): a list of layers that define the DNN architecture for the outcome model.
    - ```g_layers```(PyTorch Sequential module): a list of layers that define the DNN architecture for the selection model.
    - ```rho```(PyTorch Parameter): correlation between the error terms in the selection equation and outcome eqaution.
    - ```activation```: the activation function to use in the hidden layers (default: nn.ReLU).
    - ```dropout```: the dropout probability to use in the hidden layers (default: 0.0).
    - ```batchnorm```: whether to use batch normalization in the hidden layers (default: False).
    - ```bias```: whether to include bias in the linear layers (default: True).
   - **functions**
    - ```forward```: takes an input tensor x and returns the concatenation of the outputs of the DNNs for the outcome model (f) and the selection model (g).
    - ```forward_f```: takes an input tensor x and returns the output of the DNN for the outcome model.
    - ```forward_g```: takes an input tensor x and returns the output of the DNN for the selection model. 

#### **3.1.2. Train the model** 
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
  - ```Camelyon17Transform```: Transforms (normalization) for the Camelyon17 dataset. 
  - ```PovertyMapTransform```: Transforms (Color jittering) for the Poverty Map dataset.
  - ```RxRx1Transform```: Transforms (RandAugment) for the RxRx1 dataset.
  - ```IWildCamTransform```: Transforms (RandAugment) for the iWildCam dataset.
 2. For each transformed mini-batch of data, the model is trained using the loss function of each task (classification, regression, and multiclass classification). After the training of the model, the method calculates the loss and AUC score for the validation data using the trained model. The loss and scores (F1 score and accuracy for the classifier, and Pearson for the regressor) for both training and validation data are stored in the ```train_loss_traj```, ```valid_loss_traj```, ```train_score_traj```, and ```valid_score_traj``` lists. The code also stores the model's parameters that produce the lowest validation loss and the highest validation accuracy. 
 3. Checks if the current validation loss is lower than the best validation loss seen so far. If so, it saves the current state of the network as the best model. 
 4. The ```fit()``` returns the best model. The final trained model object is saved in [results](./results/).

#### **3.1.3. Model selection ** 

In the HeckmanDG on image data, we seperately select the best parameters of selection model (g) and outcome model (f). The selection model is selected by
data-specific evaluation metric (user can select the metric among accuracy or f1 score) and the outcome model is selected by task-specific evalauation metric (user cfan select the user can select the metric among accuracy or f1 score in the classification and mse, mae, or pearson coefficient in the regression task. All metrics are aleady set by recommended evaluation metric in ICLR paper, and user can select in [args.model_selection_metric]. After the model selection, we combine the f and g networks as one model.

In the HeckmanDG framework for image data, the selection (g) and outcome (f) models are separately optimized to select the best parameters for each model. The selection model is chosen based on a data-specific evaluation metric that the user can choose from among accuracy or F1 score. The outcome model is selected based on a task-specific evaluation metric that the user can choose from among accuracy or F1 score for classification tasks, or MSE, MAE, or Pearson coefficient for regression tasks. This repo follows the set of evaluation metrics recommended in the ICLR paper. The user can specify their desired evaluation metric using the ```args.model_selection_metric```.

Once the selection and outcome models are optimized, the ```f``` and ```g``` networks are combined into a single model.

args.model_selection_type = option 1 or option 2
- USER1: trainig doamins, testing domains (option 1-current)
- USER2: trainig domains, id-domains, ood domains, testing domains (option 2: )
 - domain 1, domain 2, domain 3, (training, id validaiotn) domain 4 (ood), domain 5 (testing)

trainig domains: 1,2,3,4
testing domains: 1,2,3,4 (internal), 5 (external)


******************** scenario 2: (W proportion of id validation): HOW Much user want to weight (0~1) : 0.5 
- trained model -> mean of internal loss (id-validation): 1 * w mean of extenal loss (ood-validation): 5  * (1-w)
validtion mean of ex and in: 3
(1000,1000,1000,6000,1000)

********** scenario 1: 
- 80 training domains
- id+ood validation
- interal validation

*********** scenario 3: dont use split
- 100% training
- ood validation


sc1: 80%/20% (training(validation)/testing) model (g&f) combined datset(ID+OOD)
sc2: 
sc3: clear but sensitive (



- **selection model (g) ** ID validation ** domain prediction
 - epoch 1 - F1: 0.5
 - epoch 2 - F1: 0.7
 - epoch 3 - F1: 0.5
 - epoch 4 - F1: 0.7
 - epoch 5 - F1: 0.9 -> best model (g_hat)

- **outcome model (f) ** OOD validation ** outfome prediction
 - epoch 1 - F1: 0.5
 - epoch 2 - F1: 0.7
 - epoch 3 - F1: 0.9 -> best model (f_hat)
 - epoch 4 - F1: 0.7
 - epoch 5 - F1: 0.5



### **4. Evaluation**
This section evaluates the trained model on the training, validation, and test data. 

- For tabular data, the internal and external validations are performed. The functiton ```model.predict_proba(x)``` yields prediction results of the trained model. The prediction performance (AUC score; using the function ```roc_auc_socre()```) for each domain in the test dataset is then calculated. They are summarized as mean scores for internal, external, and all domains . The prediction results and performances are then saved to CSV files in the [results][./results/prediction/] directory. 

- For image data, the external validation is performed. The function ```model.predict(x)``` predicts the class (multi-class, continuous labels) in the the binary classification (multiclass classification, regression). The shape of the prediction results would be as follows:
 - Binary Classification: (N, 1) 
 - Multiclass Classification: (N, J) -> In multinomial classification, the probability scores represent the probability of the observation belonging to each of the mutually exclusive classes, and the sum of the probabilities for all classes is 1.
 - Regression: (N, 1) 

- The ```prediction()``` function calculates the classificatin performances (AUC score (```roc_auc_score()```), F1 score(```f1_socre()```), and accuracy(```accuracy_score()```)) and regression performances (MSE, MAE, Pearson coefficient) for the training/validation/testing data.  The prediction results and performances are then saved to CSV files in the [results][./results/prediction/] directory. 

- For all ```data_type```, the ```plots_loss()``` generates a plot of the training loss for each domain in a different color. The ```plots_probit()``` generates a plot (histogram) of the distribution of the selection probits for each domain. This function uses ```model.get_selection_probit()``` function that can yield the probits.

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

## References
Please refer to the following papers to set hyperparameters and reproduce experiments on the WILDS benchmark.
- [WILDS](https://proceedings.mlr.press/v139/koh21a) paper: Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., ... & Liang, P. (2021, July). Wilds: A benchmark of in-the-wild distribution shifts. In International Conference on Machine Learning (pp. 5637-5664). PMLR.
- [HeckmanDG](https://openreview.net/forum?id=fk7RbGibe1) paper: Kahng, H., Do, H., & Zhong, J. Domain Generalization via Heckman-type Selection Models. In The Eleventh International Conference on Learning Representations.

* ONE FILE for all datasets (and the readme.md file)
* What are the input and output of each step?
* What is the shape of the data (image, tabular)
* what functions for what
-->

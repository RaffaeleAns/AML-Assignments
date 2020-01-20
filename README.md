<img src="https://github.com/RaffaeleAns/AML-Assignments/blob/master/images/DS%20Logo.png" width = "500">

# AML Assignments Repository
During the course of Advanced Machine Learning, I have developed 5 Assignments about different topics of Deep Learning:

## 1. FeedForward Neural Network

The assignment consists in the prediction of default payments using a feed Forward Neural Network.

The dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. 

After a brief Data Exploration, the classification problem has been solved.
Some preprocessing operations were needed to deal with unbalanced classes. 
2 models have been developed: the first baseline one has no care for the unbalanced classes, the second one uses weights in the training phase.

The results are evaluated and compared.

## 2. Autoencoders 

The assignment consists on the prediction of grayscale images of letters P - Z of handwritten letter dataset with a visual investigation of the reconstruction abilities of an auto-encoder architecture.

The Dataset consists of 14000 training labelled images and 8800 test images.

The Analysis has been conducted in 3 steps: first I have explored the dataset and made the preprocessing operations needed; second the Neural Network has been compiled and evaluated; last, the autoencoder's reconstruction abilities have been investigated and a Neural Network with encoded input has been designed, evaluated and compared with the baseline one. 

## 3. Convolutional Neural Network

The task of the assignment is the design of a CNN architecture and its training overt the MNIST digits dataset.

After the classic preprossing operations, the Convolutional Neural Network has been designed with respect to the hard constraint of a maximum 7.5 k parameters. 
The CNN, its architecture and its performances have been widely discussed in the report.

## 4. Transfer Learning

The task of the assignment is Transfer Learning using a CNN pretrained on IMAGENET.

The dataset, available on Kaggle (https://www.kaggle.com/slothkong/10-monkey-species), consists of 1300 images over 10 different monkeys species. 
The architecture used is VGG16.

The preprocessing operations include a Data Augmentation phase, in order to reduce the risk of overfitting.

The results of three specialized classification models, that uses VGG16 as feature extractor at different levels, are evaluated and compared.

## 5. Hyperparameters Optimization

The task of the assignment is Hyperparameter Optimization (HPO) of a neural network, with the aim to maximize its Accuracy on 10 fold cross validation.
The dataset to use is named "fertility", available on the "OpenML" website: https://www.openml.org/d/1473
The dataset refers to a binary classification problem and consists of around 100 instances and 9 numeric features (excluding the class).

The task consists of 2 steps:

 - Step 1: HPO for just 2 neural network's hyperparameters: learning rate and momentum. the results are compared with GridSearch and RandomSearch
 
 - Step 2: HPO for just 4 neural network's hyperparameters: learning rate, momentum and the number of units in the two hidden layers.

For the Optimization purposes SMAC3 package has been used, while the Neural Networks have been defined with the help of Scikit Learn library

The results obtained are evaluated, with a focus on the evolution of the optimization process. 

<p align = "center">
  <img src="https://github.com/RaffaeleAns/AML-Assignments/blob/master/images/AR%20Logo.png" width = "250">
</p>    
    

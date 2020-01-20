
# Hyperparameters Optimization

In this repo you can find the [Python Notebook](https://github.com/RaffaeleAns/AML-Assignments/blob/master/HPO/HPO_Script.ipynb) and a brief [report](https://github.com/RaffaeleAns/AML-Assignments/blob/master/HPO/HPO_Report.pdf).

The task of the assignment is Hyperparameter Optimization (HPO) of a neural network, with the aim to maximize its Accuracy on 10 fold cross validation.
The dataset to use is named "fertility", available on the "OpenML" website: https://www.openml.org/d/1473
The dataset refers to a binary classification problem and consists of around 100 instances and 9 numeric features (excluding the class).

The task consists of 2 steps:

 - Step 1: HPO for just 2 neural network's hyperparameters: learning rate and momentum. the results are compared with GridSearch and RandomSearch
 
 - Step 2: HPO for just 4 neural network's hyperparameters: learning rate, momentum and the number of units in the two hidden layers.

For the Optimization purposes [SMAC3](https://github.com/automl/SMAC3) package has been used, while the Neural Networks have been defined with the help of Scikit Learn library

The results obtained are evaluated, with a focus on the evolution of the optimization process. 

<p align = "center">
  <img src="https://github.com/RaffaeleAns/AML-Assignments/blob/master/images/AR%20Logo.png" width = "250">
</p>    

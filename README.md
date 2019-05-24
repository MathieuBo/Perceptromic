# Perceptromic analysis

### Reference
Machine learning reveals pathological signatures inducedby patient-derived alpha-synuclein structures. 

Bourdenx M., Nioche A. et al. *in revision*


## AIM

This analysis allows ranking of variables using Multiple Layer Perceptrons (MLPs). 

We use MLPs with a constrained architecture (3 neurons per layer - 1 hidden layer). 

A random matrix of the same size of the input matrix is used as a control of learning.

## Steps
### Overall description

1. All combinations of 3 variables from the input dataset will be use to train an MLP to predict 3 ouput variables. with a 50-folds cross validation (random sampling among available samples using a 80%/20% splitting) 
2. All combinations are then ranked regarding their performance in perdicting the 3 output variables. The top 1% is selected for further analysis.
3. The top 1% selected from the previous step is break into single variable and variables are ranked by the frequency of appearance in the top 1% of best combinations of 3 input variables. 

### Data preparation

A matrix of *n* examples (as row) and *p* variables (columns) is used as input. 
The variables used as output of the MLPs (the variables needed to be predicted should be the last 3 columns of the matrix. 

![./input_table.png](Example)

### Main script description

Using combinations.py (main folder): 

* The ``DataManager`` class will
	* Load the file (as .txt)
	* Split input and output variables
	* Center and normalize the data (Standard scoring : https://en.wikipedia.org/wiki/Standard_score)
	* Scale the data between -0.5 and 0.5

* The ``Supervisor`` class will then
	* prepare kwargs of the different combinations of input variables (including cross validation steps that need to be performed)
	* Compute the different combinations

Data are obtained as SQLite databases.

## Prerequisites

Several packages are necessary:

* Python >3
* Cython (MLP have written in cython to increase computation speed)
* Numpy
* Scipy
* Matplotlib
* sqlite3
* tqdm

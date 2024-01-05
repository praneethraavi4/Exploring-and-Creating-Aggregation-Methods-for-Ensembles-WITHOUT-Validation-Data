# Exploring-and-Creating-Aggregation-Methods-for-Ensembles-WITHOUT-Validation-Data

## Overview 

In this research work, we explore ensemble aggregation methodologies applied to the Random Forest algorithm. Random Forest, a highly acclaimed machine learning ensemble technique, is known for its exceptional predictive accuracy and adaptability. This project systematically explores the aggregation of outputs from multiple decision trees within the Random Forest ensemble, employing diverse strategies such as Majority Voting, Averaging, Weighted Averaging etc

## Table of Contents
- Libraries
- Fetching the data sets
- Loading the data
- Data Preprocessing
- Data Splitting
- Arithmetic Mean method
- Geometric Mean method
- Majority Voting method
- Tree depth based weighted Average method
- Gini Impurity based Weighted Average method

## Libraries

The following libraries are used for the project :
 - pandas
 - numpy
 - SciPy
 - Scikit-Learn
 - matplotlib <br>

If the libraries are not pre installed in the environment , please use the below commands to install them
-  pip install pandas
-  pip install numpy
-  python -m pip install scipy
-  pip install -U scikit-learn
-  python -m pip install -U matplotlib

## Fetching the data sets
OpenML is a valuable platform in the field of machine learning and data science, providing access to a vast repository of open-source datasets. These datasets, sourced from a wide range of domains and sources, offer a wealth of opportunities for researchers, data scientists, and machine learning enthusiasts to explore, experiment, and develop models. <br>

The datasets can be fetched from OpenML benchmark using the following link : <br>
https://www.openml.org/

## Loading the data

The data is expected in CSV format. The data is read using pandas as dataframe. <br>
For example , <br>
df = pd.read_csv("filepath")

## Data Preprocessing

Data preprocessing in Machine Learning is a crucial step that helps enhance the quality of data to promote the extraction of meaningful insights from the data.From the data sets which we used , all the missing values are removed with various methods like replacing with zeros or mean or standard deviance etc. The data sets are accounted for duplicates.Also , the Categorical features are converted into numerical with the help of Label Encoder.

## Data Splitting
The data is split into train and test data sets using Scikit-Learn library. The train test split used in this project is 70/30.


## Arithmetic Mean method

The tree probabilities are extracted from each tree in a Random forest classifier and the average of all tree probabilities for each class is calculated. The class with highest mean probability is the final output of the Classifier.





## Geometric Mean method 
Similar to Arithmetic mean , The tree probabilities are extracted from each tree in a Random forest classifier but here ,  the Geometric mean of all tree probabilities for each class is calculated. The class with highest Geometric mean probability is the final output of the Classifier.




## Majority Voting method
In this method , the class is chosen upon the basis of majority vote. The class which occurs most frequently as the output of individual trees is decided as the final output of the classifier



## Tree depth based weighted Average method

In this method , weights are assigned to each tree and the weighted average of tree probabilities are calculated. The class with highest weighted average is declared as the final output.
The weights are calculated as inverse to depth of the tree


## Gini Impurity based weighted Average method

In this method , weights are assigned to each tree and the weighted average of tree probabilities are calculated. The class with highest weighted average is declared as the final output.
The weights are calculated as inverse to Gini Impurity




## Running the Code

- Select the file Cardiovascular-Disease.py
- Fetch the data set ( Cardiovascular-Disease) from OpenML link specified above and save it in CSV form.
- Read the datasets as pandas dataframe as specified above and run the .py file in a python environment.

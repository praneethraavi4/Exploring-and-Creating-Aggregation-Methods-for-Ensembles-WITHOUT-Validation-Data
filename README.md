# Exploring-and-Creating-Aggregation-Methods-for-Ensembles-WITHOUT-Validation-Data

## Overview 

In this research work, we explore ensemble aggregation methodologies applied to the Random Forest algorithm. Random Forest, a highly acclaimed machine learning ensemble technique, is known for its exceptional predictive accuracy and adaptability. This project systematically explores the aggregation of outputs from multiple decision trees within the Random Forest ensemble, employing diverse strategies such as Majority Voting, Averaging, Weighted Averaging etc

## Table of Contents
- Libraries
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

## Loading the data

The data is expected in CSV format. The data is read using pandas as dataframe. <br>
For example , <br>
df = pd.read_csv("filepath")

## Data Preprocessing

Data preprocessing in Machine Learning is a crucial step that helps enhance the quality of data to promote the extraction of meaningful insights from the data.From the data sets which we used , all the missing values are removed with various methods like replacing with zeros or mean or standard deviance etc. The data sets are accounted for duplicates.Also , the Categorical features are converted into numerical with the help of Label Encoder.

## Data Splitting
The data is split into train and test data sets using Scikit-Learn library. The train test split used in this project is 70/30.
![image](https://github.com/praneethraavi4/Exploring-and-Creating-Aggregation-Methods-for-Ensembles-WITHOUT-Validation-Data/assets/135500160/10ff4f3d-9e12-469d-9e0a-956381db5767)

## Arithmetic Mean method

The tree probabilities are extracted from each tree in a Random forest classifier and the average of all tree probabilities for each class is calculated. The class with highest mean probability is the final output of the Classifier.

![image](https://github.com/praneethraavi4/Exploring-and-Creating-Aggregation-Methods-for-Ensembles-WITHOUT-Validation-Data/assets/135500160/dd62fe5c-acbd-4758-8a9d-668147ba0d37)



## Geometric Mean method 
Similar to Arithmetic mean , The tree probabilities are extracted from each tree in a Random forest classifier but here ,  the Geometric mean of all tree probabilities for each class is calculated. The class with highest Geometric mean probability is the final output of the Classifier.
![image](https://github.com/praneethraavi4/Exploring-and-Creating-Aggregation-Methods-for-Ensembles-WITHOUT-Validation-Data/assets/135500160/9d42b7f6-0c9b-46cf-bc08-9564ef999c9e)

## Majority Voting method
In this method , the class is chosen upon the basis of majority vote. The class which occurs most frequently as the output of individual trees is decided as the final output of the classifier
![image](https://github.com/praneethraavi4/Exploring-and-Creating-Aggregation-Methods-for-Ensembles-WITHOUT-Validation-Data/assets/135500160/1a8ed549-736b-403a-957c-bc7c955d68fb)

## Tree depth based weighted Average method

In this method , weights are assigned to each tree and the weighted average of tree probabilities are calculated. The class with highest weighted average is declared as the final output.
The weights are calculated as inverse to depth of the tree

![image](https://github.com/praneethraavi4/Exploring-and-Creating-Aggregation-Methods-for-Ensembles-WITHOUT-Validation-Data/assets/135500160/521ec016-57e9-4165-94b1-d2177e296f76)

## Gini Impurity based weighted Average method

In this method , weights are assigned to each tree and the weighted average of tree probabilities are calculated. The class with highest weighted average is declared as the final output.
The weights are calculated as inverse to Gini Impurity
![image](https://github.com/praneethraavi4/Exploring-and-Creating-Aggregation-Methods-for-Ensembles-WITHOUT-Validation-Data/assets/135500160/05d6e548-0317-4e2a-b163-3677224b98ac)


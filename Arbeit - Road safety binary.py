#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats.mstats import gmean
from scipy.stats import mode
from sklearn.model_selection import KFold , cross_val_score


# In[7]:


# Reading the csv data as dataframe
df = pd.read_csv("dataset_.csv")
df


# In[8]:


#Replacing the column values by 0 and 1
df['col_33'] = df['col_33'].replace({1: 0, 2: 1})


# In[12]:


# Checking for duplicates
df.duplicated().sum()


# In[14]:


# Dropping the duplicate values 
df = df.drop_duplicates(keep ="first")
df.duplicated().sum()


# In[15]:


#checking for missing values
pd.isna(df).sum()


# In[10]:


#Splitting the data into features(X) and Target variable(Y)
X = df.iloc[:,0:31]
Y = df.iloc[:,32]


# In[5]:


# Splitting the data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state= 41 )


# # Arithmetic Mean method(Default method)

# In[6]:


rf = RandomForestClassifier(n_estimators = 100,criterion='gini', min_samples_split=4, min_samples_leaf=1)
rf.fit(X_train, y_train)
predicted_labels1= rf.predict(X_test)
accuracy1 = accuracy_score(y_test, predicted_labels1)
print(accuracy1)


# In[7]:


k_folds = 5

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(rf, X, Y, cv=kf)
average_accuracy1 = np.mean(cross_val_scores)
print("Average accuracy:", average_accuracy1)
print(cross_val_scores)


# # Geometric mean method

# In[8]:


predicted_labels2 = []
geometric_mean_probabilities_list = []

epsilon = 1e-9

for i in range(len(X_test)):
    tree_probabilities = []
    for tree in rf.estimators_:
        tree_probabilities.append(tree.predict_proba(X_test.iloc[i].values.reshape(1, -1)))

    combined_probabilities = np.concatenate(tree_probabilities, axis=0)
    combined_probabilities_smoothed = combined_probabilities + epsilon
    geometric_mean_probabilities = gmean(combined_probabilities_smoothed, axis=0)
    predicted_class = np.argmax(geometric_mean_probabilities)
    predicted_labels2.append(predicted_class)

predicted_labels2 = np.array(predicted_labels2)
true_labels = y_test
accuracy2 = accuracy_score(true_labels, predicted_labels2)
print("Accuracy score:", accuracy2)


# In[9]:


# Using Cross validation

epsilon = 1e-9
rf = RandomForestClassifier(n_estimators=100)  
y_train = np.array(y_train)
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)  

accuracies2 = []

for train_index, val_index in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
    rf.fit(X_fold_train, y_fold_train)

    predicted_labels2 = []
    for i in range(len(X_fold_val)):
        tree_probabilities = []
        for tree in rf.estimators_:
            tree_probabilities.append(tree.predict_proba(X_fold_val.iloc[i].values.reshape(1, -1)))
        combined_probabilities = np.concatenate(tree_probabilities, axis=0)
        combined_probabilities_smoothed = combined_probabilities + epsilon
        geometric_mean_probabilities = gmean(combined_probabilities_smoothed, axis=0)
        predicted_class = np.argmax(geometric_mean_probabilities)
        predicted_labels2.append(predicted_class)

    predicted_labels2 = np.array(predicted_labels2)
    accuracy = accuracy_score(y_fold_val, predicted_labels2)
    accuracies2.append(accuracy)

average_accuracy2 = np.mean(accuracies2)
print("Mean cross-validated accuracy:", average_accuracy2)
print(accuracies2)


# # Majority voting method

# In[10]:


predicted_labels3 = []
for i in range(len(X_test)):
    tree_predictions = []
    for tree in rf.estimators_:  
        tree_predictions.append(np.argmax(tree.predict_proba(X_test.iloc[i].values.reshape(1, -1))))
    predicted_class, _ = mode(tree_predictions)
    predicted_labels3.append(predicted_class[0])

predicted_labels3 = np.array(predicted_labels3)
true_labels = y_test
accuracy3 = accuracy_score(true_labels, predicted_labels3)
print("Accuracy score:", accuracy3)


# In[11]:


print(accuracy3)


# In[12]:


# Using Cross validation
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np


num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=41)
accuracies3 = []

for train_index, val_index in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
    rf.fit(X_fold_train, y_fold_train)
    predicted_labels3 = []
    for i in range(len(X_fold_val)):
        tree_predictions = []
        for tree in rf.estimators_:
            tree_predictions.append(np.argmax(tree.predict_proba(X_fold_val.iloc[i].values.reshape(1, -1))))
        predicted_class, _ = mode(tree_predictions)
        predicted_labels3.append(predicted_class[0])
    predicted_labels3 = np.array(predicted_labels3)
    accuracy_fold = accuracy_score(y_fold_val, predicted_labels3)
    accuracies3.append(accuracy_fold)

average_accuracy3 = np.mean(accuracies3)
print("Mean cross-validated accuracy:", average_accuracy3)


# In[14]:


print("Mean cross-validated accuracy:", average_accuracy3)
print(accuracies3)


# # Tree Depth based weighting

# In[15]:


predicted_labels_weighted = []
weighted_mean_probabilities_list = []

tree_depths = [tree.get_depth() for tree in rf.estimators_]
max_depth = max(tree_depths)
tree_weights = [1 / (depth) for depth in tree_depths] 

for i in range(len(X_test)):
    tree_probabilities = []
    for j, tree in enumerate(rf.estimators_):
        # Multiplying each tree's probabilities by its corresponding weight
        weighted_tree_probs = tree.predict_proba(X_test.iloc[i].values.reshape(1, -1)) * tree_weights[j]
        tree_probabilities.append(weighted_tree_probs)

    # Combine the weighted probabilities
    combined_probabilities = np.sum(tree_probabilities, axis=0)

    # Calculate the weighted mean of probabilities
    weighted_mean_probabilities = np.sum(combined_probabilities, axis=0) / np.sum(tree_weights)
    predicted_class = np.argmax(weighted_mean_probabilities)
    predicted_labels_weighted.append(predicted_class)
    weighted_mean_probabilities_list.append(weighted_mean_probabilities)

predicted_labels_weighted = np.array(predicted_labels_weighted)
true_labels = y_test
accuracy4 = accuracy_score(true_labels, predicted_labels_weighted)
print("Accuracy score with depth-based weighted average:", accuracy4)


# In[20]:


def custom_weighted_accuracy(estimator, X, y):
    predicted_labels_weighted = []
    weighted_mean_probabilities_list = []

    tree_depths = [tree.get_depth() for tree in estimator.estimators_]
    max_depth = max(tree_depths)
    tree_weights = [1 / depth for depth in tree_depths]

    for i in range(len(X)):
        tree_probabilities = []
        for j, tree in enumerate(estimator.estimators_):
            # Multiplying each tree's probabilities by its corresponding weight
            weighted_tree_probs = tree.predict_proba(X.iloc[i].values.reshape(1, -1)) * tree_weights[j]
            tree_probabilities.append(weighted_tree_probs)

        # Combine the weighted probabilities
        combined_probabilities = np.sum(tree_probabilities, axis=0)

        # Calculate the weighted mean of probabilities
        weighted_mean_probabilities = np.sum(combined_probabilities, axis=0) / np.sum(tree_weights)
        predicted_class = np.argmax(weighted_mean_probabilities)
        predicted_labels_weighted.append(predicted_class)
        weighted_mean_probabilities_list.append(weighted_mean_probabilities)

    predicted_labels_weighted = np.array(predicted_labels_weighted)
    true_labels = y
    accuracy = accuracy_score(true_labels, predicted_labels_weighted)
    return accuracy

cross_val_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring=custom_weighted_accuracy)
average_accuracy4 = cross_val_scores.mean()
print("Mean Accuracy score with cross-validation:", average_accuracy4)
print(cross_val_scores)


# # Gini Impurity based weighting

# In[16]:


predicted_labels_weighted = []
weighted_mean_probabilities_list = []

tree_weights = []

for tree in rf.estimators_:
    gini_impurity = 1.0 - np.sum((tree.tree_.value[0] / np.sum(tree.tree_.value[0], axis=1, keepdims=True)) ** 2, axis=1)
    weight = 1.0 / (gini_impurity.mean()) 
    tree_weights.append(weight)

for i in range(len(X_test)):
    tree_probabilities = []
    for j, tree in enumerate(rf.estimators_):
        weighted_tree_probs = tree.predict_proba(X_test.iloc[i].values.reshape(1, -1)) * tree_weights[j]
        tree_probabilities.append(weighted_tree_probs) 
    combined_probabilities = np.sum(tree_probabilities, axis=0)
    weighted_mean_probabilities = np.sum(combined_probabilities, axis=0) / np.sum(tree_weights)
    predicted_class = np.argmax(weighted_mean_probabilities)
    predicted_labels_weighted.append(predicted_class)
    weighted_mean_probabilities_list.append(weighted_mean_probabilities)

predicted_labels_weighted = np.array(predicted_labels_weighted)
true_labels = y_test
accuracy5 = accuracy_score(true_labels, predicted_labels_weighted)
print("Accuracy score with Gini impurity-based weighted average:", accuracy5)


# In[17]:


# Using Cross validation


def calculate_gini_weights(tree):
    gini_impurity = 1.0 - np.sum((tree.tree_.value[0] / np.sum(tree.tree_.value[0], axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 / gini_impurity

def custom_weighted_accuracy(estimator, X, y):
    tree_weights = [calculate_gini_weights(tree) for tree in estimator.estimators_]
    predicted_labels_weighted = []
    for i in range(len(X)):
        tree_probabilities = []
        for j, tree in enumerate(estimator.estimators_):
            weighted_tree_probs = tree.predict_proba(X.iloc[i].values.reshape(1, -1)) * tree_weights[j]
            tree_probabilities.append(weighted_tree_probs) 
        combined_probabilities = np.sum(tree_probabilities, axis=0)
        weighted_mean_probabilities = np.sum(combined_probabilities, axis=0) / np.sum(tree_weights)
        predicted_class = np.argmax(weighted_mean_probabilities)
        predicted_labels_weighted.append(predicted_class)

    predicted_labels_weighted = np.array(predicted_labels_weighted)
    return accuracy_score(y, predicted_labels_weighted)

custom_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring=custom_weighted_accuracy)  
print("Cross-Validation Scores with Gini Impurity-based weighted average:", custom_scores)
average_accuracy5 = np.mean(custom_scores)
print("Mean CV Accuracy with Gini Impurity-based weighted average:", average_accuracy5 )


# # Comparision of Accuracy scores from different aggregation methods

# In[18]:


import matplotlib.pyplot as plt

# List of accuracies from different aggregation methods
accuracy_values = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5]

# Different aggregation methods
method_names = ['Arithmetic Mean method', 'Geometric Mean method', 'Majority Voting method', 'Tree depth based weighted average method', 'Gini Impurity based weighted average method']

# Creating a line plot
plt.figure(figsize=(10, 6))
plt.plot(method_names, accuracy_values, marker='o', linestyle='-')
plt.xlabel('Methods')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Methods')
plt.ylim(0, 1.0)  # Setting the y-axis range from 0 to 1

# Showing the plot
plt.xticks(rotation=45)  # Rotating method names for better readability
plt.tight_layout()
plt.grid(True)  # Adding grid lines
plt.show()


# In[21]:


import matplotlib.pyplot as plt

# List of accuracies from different aggregation methods
accuracy_values = [average_accuracy1, average_accuracy2, average_accuracy3, average_accuracy4, average_accuracy5]

# Different aggregation methods
method_names = ['Arithmetic Mean method', 'Geometric Mean method', 'Majority Voting method', 'Tree depth based weighted average method', 'Gini Impurity based weighted average method']

# Creating a line plot
plt.figure(figsize=(10, 6))
plt.plot(method_names, accuracy_values, marker='o', linestyle='-')
plt.xlabel('Methods')
plt.ylabel('Cross Validation Accuracy')
plt.title('Cross Validation Accuracy Comparison of Different Methods')
plt.ylim(0, 1.0)  # Setting the y-axis range from 0 to 1

# Showing the plot
plt.xticks(rotation=45)  # Rotating method names for better readability
plt.tight_layout()
plt.grid(True)  # Adding grid lines
plt.show()


# In[ ]:





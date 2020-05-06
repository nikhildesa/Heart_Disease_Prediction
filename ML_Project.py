# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:00:16 2020

@author: nikhi
"""
#<-------Libraries-------->
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
import seaborn as sns

#<---------dataset-------->
df = pd.read_csv("Data.csv")
df.head()
accuracy = dict()
print("No of rows in dataset",df.shape[0])
print("No of columns in dataset",df.shape[1])


#<--------Train Test Split------->

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

train, validate, test = train_validate_test_split(df)


X_train = train.drop(['target'], axis = 1)
y_train = train.target.values

X_validate = validate.drop(['target'], axis = 1)
y_validate = validate.target.values

X_test = test.drop(['target'], axis = 1)
y_test = test.target.values




#<--------logistic regression------->
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_validate)
cm_lr = confusion_matrix(y_validate,y_pred)


print("Accuracy Logistic Regression",accuracy_score(y_validate,y_pred))
print("Precision Logistic Regression",precision_score(y_validate,y_pred))
print("Recall Logistic Regression",recall_score(y_validate,y_pred))
print("F1 score Logistic Regression",f1_score(y_validate,y_pred))

accuracy['logistic Regression'] = accuracy_score(y_validate,y_pred)
#<-------------KNN--------------->
krange = range(1,26)
scores = []
for k in krange:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_validate)
    scores.append(accuracy_score(y_validate,y_pred))

"""
plt.plot(krange,scores)
plt.xlabel("no of k")
plt.ylabel("Accuracy")
"""
optimum_k = scores.index(max(scores))
print("Optimum k is ",optimum_k)

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_validate)
cm_knn = confusion_matrix(y_validate,y_pred)


print("Accuracy KNN",accuracy_score(y_validate,y_pred))
print("Precision KNN",precision_score(y_validate,y_pred))
print("Recall KNN",recall_score(y_validate,y_pred))
print("F1 score KNN",f1_score(y_validate,y_pred))

accuracy['KNN'] = accuracy_score(y_validate,y_pred)

#<----------------- Naive Bayes---------------->

nb = GaussianNB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_validate)
cm_nb = confusion_matrix(y_validate,y_pred)

print("Accuracy Naive Bayes",accuracy_score(y_validate,y_pred))
print("Precision Naive Bayes",precision_score(y_validate,y_pred))
print("Recall Naive Bayes",recall_score(y_validate,y_pred))
print("F1 score Naive Bayes",f1_score(y_validate,y_pred))

accuracy['Naive Bayes'] = accuracy_score(y_validate,y_pred)

#<---------------------SVM----------------------->
sv = svm.SVC()
sv.fit(X_train,y_train)
y_pred = sv.predict(X_validate)
cm_svm = confusion_matrix(y_validate,y_pred)

print("Accuracy SVM",accuracy_score(y_validate,y_pred))
print("Precision SVM",precision_score(y_validate,y_pred))
print("Recall SVM",recall_score(y_validate,y_pred))
print("F1 score SVM",f1_score(y_validate,y_pred))

accuracy['SVM'] = accuracy_score(y_validate,y_pred)

#<------------- Decision Tree--------------->

dt = tree.DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_validate)
cm_dt = confusion_matrix(y_validate,y_pred)

print("Accuracy Decision Tree",accuracy_score(y_validate,y_pred))
print("Precision Decision Tree",precision_score(y_validate,y_pred))
print("Recall Decision Tree",recall_score(y_validate,y_pred))
print("F1 score Decision Tree",f1_score(y_validate,y_pred))

accuracy['Decision Tree'] = accuracy_score(y_validate,y_pred)

#<------------------ NN --------------------->

import sklearn.neural_network

nn = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', 
                                                 alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
                                                 max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
                                                 nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                                                 n_iter_no_change=10)
nn.fit(X_train,y_train)
y_pred = nn.predict(X_validate)
cm_nn = confusion_matrix(y_validate,y_pred)
print("Accuracy NN",accuracy_score(y_validate,y_pred))

accuracy['Neural network'] = accuracy_score(y_validate,y_pred)

#<-------------- Testing -------------------->
testing_accuracy = dict()

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
cm_lr_testing = confusion_matrix(y_test,y_pred)


print("Accuracy Logistic Regression",accuracy_score(y_test,y_pred))
print("Precision Logistic Regression",precision_score(y_test,y_pred))
print("Recall Logistic Regression",recall_score(y_test,y_pred))
print("F1 score Logistic Regression",f1_score(y_test,y_pred))
testing_accuracy['Logistic Regression'] = accuracy_score(y_test,y_pred)



nb = GaussianNB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)
cm_nb_testing = confusion_matrix(y_test,y_pred)

print("Accuracy Naive Bayes",accuracy_score(y_test,y_pred))
print("Precision Naive Bayes",precision_score(y_test,y_pred))
print("Recall Naive Bayes",recall_score(y_test,y_pred))
print("F1 score Naive Bayes",f1_score(y_test,y_pred))
testing_accuracy['Naive Bayes'] = accuracy_score(y_test,y_pred)




model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', 
                                                 alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
                                                 max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
                                                 nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                                                 n_iter_no_change=10)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
cm_nn_testing = confusion_matrix(y_test,y_pred)
print("Accuracy NN",accuracy_score(y_test,y_pred))
testing_accuracy['NN'] = accuracy_score(y_test,y_pred)

#<----------- Comparing models--------->

plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
color = ["Red","Green","Blue"]
sns.set_style("whitegrid")
plt.figure(figsize=(10,5))

plt.ylim(0,1,0.1)
sns.barplot(x=list(testing_accuracy.keys()), y=list(testing_accuracy.values()), palette=color)



#<---------------- comparing testing models---------------->

plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
color = ["Red","Green","Blue"]
sns.set_style("whitegrid")
plt.figure(figsize=(10,5))

plt.ylim(0,1,0.1)

sns.barplot(x=list(testing_accuracy.keys()), y=list(testing_accuracy.values()), palette=color)

plt.subplot(2,3,1)
plt.title("Logistic Regression")
sns.heatmap(cm_lr_testing,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,2)
plt.title("Naive Bayes")
sns.heatmap(cm_nb_testing,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,3)
plt.title("NN")
sns.heatmap(cm_nn_testing,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})















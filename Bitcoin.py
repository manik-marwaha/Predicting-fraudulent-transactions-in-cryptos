#!/usr/bin/env python
# coding: utf-8

/************************************************************************************
SUBJECT:            APPLIED MACHINE LEARNING
LEVEL  :            POSTGRADUATE
NAME   :            MANIK MARWAHA
UNI ID :            a1797063
PROJECT:            Predicting fraudulent transactions incryptocurrency trading
**************************************************************************************/

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split, cross_validate,StratifiedKFold
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
#from sklearn.pipeline import Pipel
import pandas as pd
import numpy as np
import ast
import string
import nltk
import matplotlib.pyplot as plt
#nltk.download('wordnet')
import seaborn as sns
from sklearn import svm
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from google.colab import drive
drive.mount('/content/gdrive')

import os
os.chdir('/content/gdrive/My Drive/Bitcoin')

"""Importing Data Files"""

df_features = pd.read_csv("elliptic_txs_features.csv")
df_edgelist = pd.read_csv("elliptic_txs_edgelist.csv")
df_classes =  pd.read_csv("elliptic_txs_classes.csv")

df_features=pd.DataFrame(df_features.values, columns = ["Feature {}".format(i) for i in range(df_features.shape[1])])
df_features

boolean =df_features['Feature 0'].duplicated().any()
boolean

"""Data distribution across all the features"""

from pylab import rcParams
rcParams['figure.figsize'] = 20, 10
#df_temp=df_features.drop("label",axis=1)
df_temp=df_features.drop("Feature 0",axis=1)
plt.xticks(rotation=90)
sns.boxplot(data=pd.melt(df_temp),x="variable", y="value")
plt.savefig("aalinone")

from pylab import rcParams
rcParams['figure.figsize'] = 15,10

"""Plot 10 features at a time"""

#df_temp=df_features.drop("label",axis=1)
#df_temp=df_temp.drop("Feature 0",axis=1)
for i in range(0,166,10):
  if i!=160:
    sns.boxplot(data=pd.melt(df_temp.iloc[:,i:i+10]),x="variable", y="value")
  else:
    sns.boxplot(data=pd.melt(df_temp.iloc[:,i:i+6]),x="variable", y="value")
  plt.xticks(rotation=75)
  name="plots/"+str(i)
  #plt.savefig(name)
  plt.show()

df_classes =  pd.read_csv("elliptic_txs_classes.csv")
df_classes=df_classes.replace("unknown","3")
df_classes = df_classes.iloc[1:]
plt.figure(figsize=(10,10))
plt.xlabel("Classes")
plt.ylabel("Number of Samples")
hist = df_classes['class'].hist()
plt.savefig("Class distribution2")

df_features["label"]=df_classes[["class"]]

"""Labelling the data"""

df_unlabelled=df_features.loc[df_features['label'] == "3"]

df_labelled=df_features.loc[df_features['label'].isin(["1","2"])]

df_labelled["label"].replace({"1": 1, "2": 2}, inplace=True)

df_labelled

"""Illicit Vs licit comparision at each timeframe"""

x=[]
y=[]
for name,group in df_labelled.groupby(["1"]):
  ilicit = (group.label == 1).sum()
  licit = (group.label == 2).sum()
  x.append(name)
  y.append(ilicit/licit)
plt.plot(x,y)
plt.xlabel("Timeframe")
plt.ylabel("illicit/licit")
plt.savefig("ilicitvslicit")
plt.show()

"""Distribution of licit vS illicit at each timeframe"""

sns.countplot(data=df_labelled,x=df_labelled.columns[1],hue="label")
plt.xlabel("Timeframe")
plt.ylabel("Frequency")
plt.savefig("Distribution")
plt.show()

"""TSNE"""

import seaborn as sns
from sklearn.manifold import TSNE
#d2.insert(100, "100", output, True) 
#print(d2)
df_subset=pd.DataFrame()
d2=df_labelled
y=d2[["label"]]
d2 = d2.drop("label",axis=1)

tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df_labelled)
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
df_subset["y"]=y
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    data=df_subset,
    legend="full",
    alpha=0.3
)
plt.savefig("tsne")
plt.show()

"""PCA"""

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(d2)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, y], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ["1","2"]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf.iloc[:,2] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)


ax.grid()
plt.savefig("pca")
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(d2, y, test_size=0.5)

"""Baseline Baseline"""

from sklearn import svm
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score
#Create a svm Classifier
clf = svm.SVC(kernel='rbf',C=5)# Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
metrics.plot_roc_curve(clf, X_test, y_test)  # doctest: +SKIP
plt.show()  
roc_auc_score(y_test, y_pred)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, annot_kws={"size": 14}, fmt='g')
plt.savefig("rf_heat.png")

"""Naive Stratified Sampling"""

from sklearn.model_selection import train_test_split
df_1=df_labelled.loc[df_labelled['label'] == 1]
y_1 = df_1[["label"]]
df_1=df_1.drop('label',axis=1)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(df_1, y_1, test_size=0.2)

df_2=df_labelled.loc[df_labelled['label'] == 2]
y_2 = df_2[["label"]]
df_2=df_2.drop("label",axis=1)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(df_2, y_2, test_size=0.9)

train_trying_x_1 = X_train_1.append(X_train_2)
train_trying_y_1 = y_train_1.append(y_train_2)
test_trying_x_1 = X_test_1.append(X_test_2)
test_trying_y_1 = y_test_1.append(y_test_2)

"""Baseline with naive stratified sampling"""

get_ipython().system('pip install lazypredict')

import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(train_trying_x_1, test_trying_x_1, train_trying_y_1, test_trying_y_1)
models

"""Baseline Logistic Regression"""

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(train_trying_x_1, train_trying_y_1)
y_pred = clf.predict(test_trying_x_1)
print(confusion_matrix(test_trying_y_1, y_pred))
print(classification_report(test_trying_y_1, y_pred))
metrics.plot_roc_curve(clf, test_trying_x_1, test_trying_y_1)
plt.savefig("roc_log")  # doctest: +SKIP
plt.show()  
roc_auc_score(test_trying_y_1, y_pred) 
sns.heatmap(confusion_matrix(test_trying_y_1, y_pred), annot=True, annot_kws={"size": 14}, fmt='g')
plt.savefig("logistic.png")

"""SVM"""

from sklearn import svm
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score
#Create a svm Classifier
clf = svm.SVC(kernel='rbf',C=5)# Linear Kernel

#Train the model using the training sets
clf.fit(train_trying_x_1, train_trying_y_1)

#Predict the response for test dataset
y_pred = clf.predict(test_trying_x_1)
print(confusion_matrix(test_trying_y_1, y_pred))
print(classification_report(test_trying_y_1, y_pred))
metrics.plot_roc_curve(clf, test_trying_x_1, test_trying_y_1)  # doctest: +SKIP
plt.savefig("roc_svc")
plt.show()  
roc_auc_score(test_trying_y_1, y_pred)
sns.heatmap(confusion_matrix(test_trying_y_1, y_pred), annot=True, annot_kws={"size": 14}, fmt='g')
plt.savefig("svc.png")

"""Random Forest"""

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=1000, random_state=0).fit(train_trying_x_1, train_trying_y_1)
y_pred = clf.predict(test_trying_x_1)
print(confusion_matrix(test_trying_y_1, y_pred))
print(classification_report(test_trying_y_1, y_pred))
metrics.plot_roc_curve(clf, test_trying_x_1, test_trying_y_1)  # doctest: +SKIP
plt.savefig("roc_rf")
plt.show()  
roc_auc_score(test_trying_y_1, y_pred)
sns.heatmap(confusion_matrix(test_trying_y_1, y_pred), annot=True, annot_kws={"size": 14}, fmt='g')
plt.savefig("rf.png")

type(test_trying_x_1)

"""Kfold for random forest and SVM"""

from sklearn.model_selection import GridSearchCV 
param_grid = {'C': [0.1,1,5,10],  
              'gamma': [1, 0.1, 0.01,"scale"], 
              'kernel': ['rbf']}  

train_trying_y_1=train_trying_y_1
test_trying_y_1 = test_trying_y_1
grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 0,scoring="f1",cv=10) 
  
# fitting the model for grid search 
grid.fit(train_trying_x_1, train_trying_y_1) 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 
y_pred = grid.predict(test_trying_x_1)
print(confusion_matrix(test_trying_y_1, y_pred))
print(classification_report(test_trying_y_1, y_pred))
metrics.plot_roc_curve(grid, test_trying_x_1, test_trying_y_1)  # doctest: +SKIP
plt.savefig("svm_grid.png")
plt.show()
sns.heatmap(confusion_matrix(test_trying_y_1, y_pred), annot=True, annot_kws={"size": 14}, fmt='g')
plt.savefig("svm_heat_grid.png")

plt.show()
roc_auc_score(test_trying_y_1, y_pred)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
param_grid = [
{'n_estimators': [10, 25,50,100], 'max_features': [5, 10,15,20,100], 
 'max_depth': [10,100,None]}
]

train_trying_y_1=train_trying_y_1
test_trying_y_1 = test_trying_y_1
forest = RandomForestClassifier()
h_forest = GridSearchCV(forest, param_grid, cv=5, scoring='f1',verbose=0)
h_forest.fit(train_trying_x_1, train_trying_y_1)
print(h_forest.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(h_forest.best_estimator_) 
y_pred = h_forest.predict(test_trying_x_1)
print(confusion_matrix(test_trying_y_1, y_pred))
print(classification_report(test_trying_y_1, y_pred))
metrics.plot_roc_curve(h_forest, test_trying_x_1, test_trying_y_1)  # doctest: +SKIP
plt.savefig("rf_grid.png")
plt.show()
sns.heatmap(confusion_matrix(test_trying_y_1, y_pred), annot=True, annot_kws={"size": 14}, fmt='g')
plt.savefig("rf_heat_grid.png")

feat_importances = pd.Series(clf.feature_importances_, index=df_1.columns)
feat_importances.nlargest(15).plot(kind='barh')
#plt.saveplot("featureimp")

"""**Preprocessing**"""

print(y)
d2

std = StandardScaler()
X_preprocessed = std.fit_transform(d2)
scaled_features_df = pd.DataFrame(X_preprocessed, index=d2.index, columns=d2.columns)

corr = scaled_features_df.corr()
f = plt.figure(figsize=(19, 15))
plt.matshow(corr, fignum=f.number)
plt.xticks(range(scaled_features_df.shape[1]), scaled_features_df.columns, fontsize=14, rotation=45)
plt.yticks(range(scaled_features_df.shape[1]), scaled_features_df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
plt.savefig("Correlation plot")

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print(to_drop)

df_preprocessed = scaled_features_df.drop(scaled_features_df[to_drop], axis=1)

df_preprocessed

from pylab import rcParams
rcParams['figure.figsize'] = 20, 10
#df_temp=df_features.drop("label",axis=1)
df_temp=df_preprocessed.drop("Feature 0",axis=1)
plt.xticks(rotation=90)
sns.boxplot(data=pd.melt(df_temp),x="variable", y="value")
plt.savefig("Updated Boxplot")

from sklearn.model_selection import train_test_split
df_labelled = pd.concat([df_preprocessed, y], axis=1)
df_1=df_labelled.loc[df_labelled['label'] == "1"]
y_1 = df_1[["label"]]
df_1=df_1.drop('label',axis=1)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(df_1, y_1, test_size=0.2)

df_2=df_labelled.loc[df_labelled['label'] == "2"]
y_2 = df_2[["label"]]
df_2=df_2.drop("label",axis=1)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(df_2, y_2, test_size=0.9)

train_trying_x_1 = X_train_1.append(X_train_2)
train_trying_y_1 = y_train_1.append(y_train_2)
test_trying_x_1 = X_test_1.append(X_test_2)
test_trying_y_1 = y_test_1.append(y_test_2)

from sklearn.svm import SVC
clf = SVC().fit(train_trying_x_1, train_trying_y_1)
plt.figure(figsize=(7,7))
y_pred = clf.predict(test_trying_x_1)
print(confusion_matrix(test_trying_y_1, y_pred))
print(classification_report(test_trying_y_1, y_pred))
metrics.plot_roc_curve(clf, test_trying_x_1, test_trying_y_1)  # doctest: +SKIP
plt.savefig("roc_rf2.png")
plt.show()  
plt.figure(figsize=(7,7))
roc_auc_score(test_trying_y_1, y_pred)
sns.heatmap(confusion_matrix(test_trying_y_1, y_pred), annot=True, annot_kws={"size": 14}, fmt='g')
plt.savefig("rf2.png")

from google.colab import files
uploaded = files.upload()

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn import tree


df = pd.read_csv("mlq5.csv")



# from sklearn.preprocessing import LabelEncoder
# Le = LabelEncoder()

# df['Outlook'] = Le.fit_transform(df['Outlook'])
# df['Climate'] = Le.fit_transform(df['Climate'])
# df['Humidity'] = Le.fit_transform(df['Humidity'])
# df['Wind'] = Le.fit_transform(df['Wind'])
# df['PlayMatch'] = Le.fit_transform(df['PlayMatch'])

y = df['PlayMatch']
df = df.drop(['PlayMatch'], axis=1)
df = pd.get_dummies(df)

X_train = df.iloc[:7,:]
X_test = df.iloc[8:,:]

y_train = y.iloc[:7]
y_test = y.iloc[8:]

print(X_train.head())

clf = DecisionTreeClassifier(random_state=1234)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_pred, y_test))
print(accuracy_score(clf.predict(X_train), y_train))

import graphviz 
dot_data = tree.export_graphviz(clf, feature_names=df.columns ,out_file=None) 
graph = graphviz.Source(dot_data) 

graph
# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(clf, 
#                    feature_names=X.columns,  
#                    class_names=y.columns,
#                    filled=True)


# print(df[(df['Outlook']!="Overcast") & (df['PlayMatch']=="No")].shape[0])

# print(df[((df['Outlook']!="Rainy") & (df['Outlook']!="Sunny")) & (df['PlayMatch']=="No")].shape[0])

"""Installing Deepwalk"""

get_ipython().system('git clone https://github.com/phanein/deepwalk.git')

os.chdir('/content/gdrive/My Drive/Bitcoin/deepwalk/') 
get_ipython().system('pip install -r requirements.txt')
get_ipython().system('python setup.py install')

"""Running deepwalk on edgelist"""

os.chdir('/content/gdrive/My Drive/Bitcoin/')
get_ipython().system('deepwalk --input new_edgelist --format edgelist --output abc')

"""Seperating node and features"""

with open("abc") as fp:
  fp=fp.read().splitlines()
t={x[:x.index(" ")]:x[x.index(" ")+1:] for x in fp[1:]}

fp[1]

t

"""matching node id with labelled data"""

x=[]
y=[]
for i in t:
  if len(df_classes[df_classes["txId"]==int(i)])!=0:
    #for j in df_labelled[df_labelled.columns[0]]:
    x.append(t[i])
    y.append(df_classes[df_classes["txId"]==int(i)]["class"])
    #print("dvvxxxads")

"""Naive Stratifies Sampling and converting embedding string to list of floats"""

tx_1=[]
ty_1=[]
tx_2=[]
ty_2=[]
for n,i in enumerate(y):
  if i.values[0]=="1":
    ttttt=[]
    for k in x[n].split(" "):
      ttttt.append(float(k))
    tx_1.append(np.array(ttttt))
    ty_1.append([int(i.values[0])])
  elif i.values[0]=="2":
    ttttt=[]
    for k in x[n].split(" "):
      ttttt.append(float(k))
    tx_2.append(np.array(ttttt))
    ty_2.append([int(i.values[0])])

len(tx_2),len(df_labelled)

from sklearn.model_selection import train_test_split

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(tx_1, ty_1, test_size=0.2)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(tx_2, ty_2, test_size=0.9)

train_trying_x_1 =[]
train_trying_y_1 =[]
test_trying_x_1  = []
test_trying_y_1  = [] 

for i in X_train_1:
  train_trying_x_1.append(i)
for i in X_train_2:
  train_trying_x_1.append(i)
for i in X_test_1:
  test_trying_x_1.append(i)
for i in X_test_2:
  test_trying_x_1.append(i)

for i in y_train_1:
  train_trying_y_1.append(i)
for i in y_train_2:
  train_trying_y_1.append(i)
for i in y_test_1:
  test_trying_y_1.append(i)
for i in y_test_2:
  test_trying_y_1.append(i)

X_train_1[0]

np.array(y_train_1).shape,np.array(X_train_1).shape

"""Random Forest, SVM and their KFold versions"""

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=1000, random_state=0).fit(train_trying_x_1, train_trying_y_1)
y_pred = clf.predict(test_trying_x_1)
print(confusion_matrix(test_trying_y_1, y_pred))
print(classification_report(test_trying_y_1, y_pred))
metrics.plot_roc_curve(clf, test_trying_x_1, test_trying_y_1)  # doctest: +SKIP
plt.savefig("roc_rf_dw")
plt.show()  
roc_auc_score(test_trying_y_1, y_pred)
sns.heatmap(confusion_matrix(test_trying_y_1, y_pred), annot=True, annot_kws={"size": 14}, fmt='g')
plt.savefig("rf_dw.png")

from sklearn.model_selection import GridSearchCV 
param_grid = {'C': [0.001,0.01,0.1,1,5,10,100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001,10,20,"scale"], 
              'kernel': ['rbf']}  

train_trying_y_1=train_trying_y_1
test_trying_y_1 = test_trying_y_1
grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 0,scoring="f1",cv=10) 
  
# fitting the model for grid search 
grid.fit(train_trying_x_1, train_trying_y_1) 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 
y_pred = grid.predict(test_trying_x_1)
print(confusion_matrix(test_trying_y_1, y_pred))
print(classification_report(test_trying_y_1, y_pred))
metrics.plot_roc_curve(grid, test_trying_x_1, test_trying_y_1)  # doctest: +SKIP
plt.savefig("svm_grid_dw.png")
plt.show()
sns.heatmap(confusion_matrix(test_trying_y_1, y_pred), annot=True, annot_kws={"size": 14}, fmt='g')
plt.savefig("svm_heat_grid_dw.png")

plt.show()
roc_auc_score(test_trying_y_1, y_pred)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
param_grid = [
{'n_estimators': [10, 25,50,100], 'max_features': [5, 10,15,20,100], 
 'max_depth': [10,100,None]}
]

train_trying_y_1=train_trying_y_1
test_trying_y_1 = test_trying_y_1
forest = RandomForestClassifier()
h_forest = GridSearchCV(forest, param_grid, cv=5, scoring='f1',verbose=0)
h_forest.fit(train_trying_x_1, train_trying_y_1)
print(h_forest.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(h_forest.best_estimator_) 
y_pred = h_forest.predict(test_trying_x_1)
print(confusion_matrix(test_trying_y_1, y_pred))
print(classification_report(test_trying_y_1, y_pred))
metrics.plot_roc_curve(h_forest, test_trying_x_1, test_trying_y_1)  # doctest: +SKIP
plt.savefig("rf_grid_dw.png")
plt.show()
sns.heatmap(confusion_matrix(test_trying_y_1, y_pred), annot=True, annot_kws={"size": 14}, fmt='g')
plt.savefig("rf_heat_grid_dw.png")


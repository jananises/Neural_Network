# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:39:34 2020

@author: Janani
"""

#Build a Model for the strength of concrete data using Neural Networks – regression Problem

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loading the dataset
#For google colab
#concrete_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Deep Learning/Data Sets/concrete.csv')
concrete_data = pd.read_csv('C:\\Data Science\\Assignments\\Neural Networks\\concrete.csv')

concrete_data.head()

print(concrete_data.shape)
print(concrete_data.columns)

#Get Statistics
concrete_data.describe()

#Check for any missing values
concrete_data.isnull().any()


#Data Visualization, plot Heatmap
corr = concrete_data.corr()
#Plot
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()


#Data Visualization, pairplot
sns.pairplot(concrete_data, diag_kind="kde")


#Changing pandas dataframe to numpy array
X = concrete_data.iloc[:,0:8].values
Y = concrete_data.iloc[:,8].values
print(X)
print(Y)
print(X.shape)

#Feature Scaling
# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
from numpy import set_printoptions
# separate array into input and output components
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:8,:])

#Splitting the dataset into training and test data-set
# Evaluate using a train and a test set
from sklearn.model_selection import train_test_split
test_size = 0.30
seed = 10
X_train, X_test, Y_train, Y_test = train_test_split(rescaledX, Y, test_size=test_size, random_state=seed)
print(X_train)
print(Y_train)


#Model Building - ANN
#Load Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras import backend

#Defining Root Mean Square Error As Metric Function 
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

#Model Building
#Model Creation - ANN - Model 1
model=Sequential()

# The Input Layer :
model.add(Dense(64,input_dim = 8,activation="relu"))

# The Hidden Layers :
model.add(Dense(32,activation="relu"))
model.add(keras.layers.normalization.BatchNormalization())

# The Output Layer :
model.add(Dense(1, activation='linear'))

#Define Optimizer
opt = keras.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999,amsgrad=False)
#Compile my model
model.compile(loss="mean_squared_error",optimizer=opt, metrics=[rmse])

model.summary()

###############################
#Model 2
#Let’s create an additional hidden layer
#Model Creation - ANN - Step2
#model=Sequential()
# The Input Layer :
#model.add(Dense(64,input_dim = 8,activation="relu"))
# The Hidden Layers :
#model.add(Dense(32,activation="relu"))
#model.add(Dense(32,activation="relu"))
#model.add(keras.layers.normalization.BatchNormalization())
# The Output Layer :
#model.add(Dense(1, activation='linear'))
#Define Optimizer
#opt = keras.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999,amsgrad=False)
#Compile my model
#model.compile(loss="mean_squared_error",optimizer=opt, metrics=[rmse])

#model.summary()

##########################################

history = model.fit(X_train,Y_train,epochs = 100 ,batch_size=32,validation_split=0.15)
#history = model.fit(X_train,Y_train,epochs = 400 ,batch_size=32,validation_split=0.15)
print(model.summary())

##Step 2 - Increase epochs to 400
#history = model.fit(X_train,Y_train,epochs = 400 ,batch_size=32,validation_split=0.15)
#print(model.summary())

#Predict and finding Rsquare score
y_predict = model.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(Y_test,y_predict))


# Plotting Loss And Root Mean Square Error For both Training And Test Sets
plt.plot(history.history['rmse'])
plt.plot(history.history['val_rmse'])
plt.title('Root Mean Squared Error')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('4.png')
plt.show()


# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:39:04 2020

@author: Janani
"""
#Build a Neural Network model for 50_startups data to predict profit 

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#startup_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Deep Learning/Data Sets/50_Startups.csv')
startup_data = pd.read_csv('C:\\Data Science\\Assignments\\Neural Networks\\50_Startups.csv')
startup_data.head()


startup_data.shape
startup_data.columns
startup_data.info()


#Clean the data
startup_data.isna().sum()


#Get Unique values of categorical variable - State
print(startup_data.State.unique())
# Now we classify them as numbers instead of their names.
startup_data['State'] = startup_data['State'].map({'New York': 0, 'California': 1, 
                                                       'Florida': 2}).astype(float)

startup_data.describe()

startup_data.head()

#Data Visualization, plot Heatmap
corr = startup_data.corr()
#Plot
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()


#Data Visualization, pairplot
sns.pairplot(startup_data, diag_kind="kde")


#Split into train and test
train_dataset = startup_data.sample(frac=0.8, random_state=0)
test_dataset = startup_data.drop(train_dataset.index)
print(test_dataset)

#Split features from labels
train_labels = train_dataset.pop('Profit')
test_labels = test_dataset.pop('Profit')
print(train_labels)
print(test_labels)


#Feature Scaling
# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
from numpy import set_printoptions
# separate array into input and output components
scaler = StandardScaler()
train_dataset = scaler.fit_transform(train_dataset)
test_dataset = scaler.transform(test_dataset)
# summarize transformed data
set_printoptions(precision=3)
print(train_dataset)
print(test_dataset)


#Model Building - ANN
#Load Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras import backend


#Defining Root Mean Square Error As our Metric Function 
def rmse(y_true, y_pred):
  return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


#Model Creation - ANN - Model 1
model=Sequential()
# The Input Layer :
model.add(Dense(12,input_dim = 4,kernel_initializer="normal", activation="relu"))
# The Hidden Layers :
model.add(Dense(8,kernel_initializer="normal", activation="relu"))
#model.add(keras.layers.normalization.BatchNormalization())
# The Output Layer :
model.add(Dense(1, kernel_initializer="normal", activation='linear'))
#Define Optimizer
opt = keras.optimizers.Adam(lr=0.1,beta_1=0.9, beta_2=0.999,amsgrad=False)
#Compile my model
model.compile(loss="mean_squared_error",optimizer=opt, metrics=[rmse])

model.summary()


#######################################################################################
#Model 2 - Creating another Hidden Layer

#Model Creation - ANN - Model 2 - Adding one more hidden layer
#model=Sequential()
# The Input Layer :
#model.add(Dense(12,input_dim = 4,kernel_initializer="normal", activation="relu"))
# The Hidden Layers :
#model.add(Dense(8,kernel_initializer="normal", activation="relu"))
#model.add(Dense(8,kernel_initializer="normal", activation="relu"))
# The Output Layer :
#model.add(Dense(1, kernel_initializer="normal", activation='linear'))
#Define Optimizer
#opt = keras.optimizers.Adam(lr=0.1,beta_1=0.9, beta_2=0.999,amsgrad=False)
#Compile my model
#model.compile(loss="mean_squared_error",optimizer=opt, metrics=[rmse])
#model.summary()
# Observation - Overfitting Problem - by increasing one more layer, error increases

#history = model.fit(X_train,Y_train,epochs = 100 ,batch_size=32,validation_split=0.15)
#Increase no of epochs
history = model.fit(train_dataset,train_labels,epochs = 400 ,batch_size=32,validation_split=0.1)
#history = model.fit(train_dataset,train_labels,epochs = 600 ,batch_size=32,validation_split=0.1)

print(model.summary())


#Predict and finding Rsquare score
y_predict = model.predict(test_dataset)

from sklearn.metrics import r2_score
print(r2_score(test_labels,y_predict))

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


#########Test Predictions###############################

#Test the predictions through Plot
test_predictions = model.predict(test_dataset).flatten()
plt.scatter(test_labels,test_predictions)
plt.xlabel('True Values [Profit]')
plt.ylabel('Predictions [Profit]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_= plt.plot([-100,100],[-100,100], 'r')

#Error Plot
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [Profit]")
_ = plt.ylabel("Count")






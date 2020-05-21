# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:33:00 2020

@author: Janani
"""
#Forest Fire Classification of area using Neural Networks
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#Loading the datasets
#forest_fire=pd.read_csv('/content/drive/My Drive/Colab Notebooks/Deep Learning/Data Sets/forestfires.csv')
forest_fire=pd.read_csv('C:\\Data Science\\Assignments\\Neural Networks\\concrete\\forestfires.csv')
forest_fire.head()

forest_fire.columns.values

forest_fire.shape

forest_fire.info()

forest_data = forest_fire
#Remove Extra Columns
forest_fire = forest_data.drop(['dayfri','daymon','daysat','daysun','daythu','daytue','daywed','monthapr','monthaug','monthdec','monthfeb',
                      'monthjan','monthjul','monthjun','monthmar','monthmay','monthnov',
                      'monthoct','monthsep'],axis=1)


forest_fire.shape

#Data Pre-Processing
# Encode Data for Month and Day
forest_fire.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
forest_fire.day.replace(('mon','tue','wed','thu','fri','sat','sun'),(1,2,3,4,5,6,7), inplace=True)


forest_fire.head()


#Check for any missing values
#forest_fire.isna().sum()
forest_fire.isnull().any()


#Descriptive Statistics
forest_fire.describe()


#let's see how many unique categories we have in this output class - size category
size_set = set(forest_fire['size_category'])
print(size_set)


# Just transforming size category to category 1 and 0.
forest_fire['size_category'] = forest_fire['size_category'].map({'small': 0, 'large': 1}).astype(int)


forest_fire.tail()

#Data Visualization, plot Heatmap
corr = forest_fire.corr()
#Plot
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()


#correlation matrix
#Correlation Matrix for Data Analysis
print("Correlation:", forest_fire.corr(method='pearson'))

#Data Visualization, pairplot
sns.pairplot(forest_fire, diag_kind="kde") 

plt.hist(forest_fire.area, bins=25)
#Observation Most of the dataset's samples fall between 0 and 300 of 'Area' 

#Split into train and test
train_dataset = forest_fire.sample(frac=0.8, random_state=0)
test_dataset = forest_fire.drop(train_dataset.index)

print(test_dataset)

#Split features from labels
train_labels = train_dataset.pop('size_category')
test_labels = test_dataset.pop('size_category')
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


#Model Creation - ANN - Model 1
# baseline model
#def create_baseline():
model=Sequential()
# The Input Layer :
model.add(Dense(22,input_dim = 11, activation="relu"))
# The Hidden Layers :
model.add(Dense(44,activation="relu"))
# The Output Layer :
model.add(Dense(1, activation='sigmoid'))
#Define Optimizer
opt = keras.optimizers.Adam(lr=0.01,beta_1=0.9, beta_2=0.999,amsgrad=False)
#Compile my model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
 # return model

#Output of this Model - for epochs = 400
#Confusion Matrix: [[76  0]
#                   [ 6 21]]
#Test Accuracy: 94.17

#Model Creation - ANN - Model 2 - Adding one more hidden layer
model=Sequential()
# The Input Layer :
model.add(Dense(11,input_dim = 11, activation="relu"))
# The Hidden Layers :
model.add(Dense(22,activation="relu"))
model.add(Dense(22,activation="relu"))
# The Output Layer :
model.add(Dense(1, activation='sigmoid'))
#Define Optimizer
opt = keras.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999,amsgrad=False)
#Compile my model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
 # return model


model.summary()

history = model.fit(train_dataset,train_labels,epochs = 400 ,batch_size=40,validation_split=0.1)
#history = model.fit(train_dataset,train_labels,epochs = 100 ,batch_size=10,validation_split=0.1)
#history = model.fit(X_train,Y_train,epochs = 100 ,batch_size=32,validation_split=0.15)
print(model.summary())


# evaluate the keras model
score = model.evaluate(test_dataset, test_labels, verbose=1)
print('Test Accuracy: %.2f' % (score[1]*100))
print('Loss: %.2f' % (score[0]))


# make probability predictions with the model
predictions = model.predict_classes(test_dataset)

from sklearn.metrics import confusion_matrix
cn=confusion_matrix(test_labels,predictions)
print("Confusion Matrix:", cn)
print('Test Accuracy: %.2f' % (score[1]*100))



# Compare predictions with actual values for the first few items in our test dataset
num_predictions = 50
diff = 0

for i in range(num_predictions):
    val = predictions[i]
    print('Predicted: ', val[0], 'Actual: ', test_labels.iloc[i], '\n')
    diff += abs(val[0] - test_labels.iloc[i])
    
    
# Plotting Loss And Root Mean Square Error For both Training And Test Sets
loss_train = plt.plot(history.history['accuracy'])
loss_val = plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('4.png')
plt.show()




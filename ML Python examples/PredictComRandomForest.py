import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from JobDataExtraction import JobDataExtraction
from os import listdir, walk, getcwd
from os.path import isfile, join
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import svm
from sklearn import linear_model

seed = 128  
rng = np.random.RandomState(seed)

def batch_creator(batch_size, dataset_length, dataset_name, nParam, train_x, train_y):
    #"""Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, nParam)
    #batch_x = preproc(batch_x)
    
    if dataset_name == 'train':
        batch_y = train_y[batch_mask] 
    return batch_x, batch_y

def ArrayShuffle(vect):
    for i in range(len(vect)):
        r=random.randint(0,i)
        swap = vect[i]
        vect[i] = vect[r]
        vect[r] = swap
    return vect

### Load training examples
ExtractObj = JobDataExtraction()        
        
# Get all Job data of interest
DataDir =  'C:\\Users\\SCatheline\\OneDrive - Schlumberger\\Testing Project\\MUZIC\\Nahomi Project\\Completed'

SubDir = [f for f in listdir(DataDir) if not isfile(join(DataDir, f))]
for DiR in SubDir:
    ExtractObj.AddJob(DataDir+ '\\' +DiR)

#Add communication data from the MuzicJob Analysis tool
ComDir = "C:\\Users\\SCatheline\\OneDrive - Schlumberger\\Testing Project\\MUZIC\Matlab code\\Statistical Analysis\\JobDataExtraction\\JobData"

SubDir = [f for f in listdir(ComDir) if not isfile(join(ComDir, f))]
for DiR in SubDir:
    ExtractObj.AddComData(ComDir+ '\\' +DiR) 

ExtractObj.CreateDictTraining()

ExtractObj.CleanTrainingEx()

FileCompress = 'C:\\Users\\SCatheline\\OneDrive - Schlumberger\\Testing Project\\MUZIC\\Nahomi Project\\Completed\\ToolCompression.txt'
ExtractObj.CompressFeatures(FileCompress)

FeatureMat = ExtractObj.FeatureTrainMat
OutputVect = ExtractObj.OutputTrainVect

## Rescale data so that the max of each column is 1 (max = 1)
for i in range(FeatureMat.shape[1]):
    FeatureMat[:,i] = FeatureMat[:,i]/max(abs(FeatureMat[:,i]))
 
nParam = FeatureMat.shape[1]
train_x = FeatureMat
train_y = OutputVect

### Separate training and validationb data
#Shuffle the exemple vector
shuffle = np.linspace(0,train_x.shape[0]-1,train_x.shape[0],dtype=int)

shuffle = ArrayShuffle(shuffle)
    
split_size = int(train_x.shape[0]*0.70)

train_x, val_x = train_x[shuffle[:split_size]][:], train_x[shuffle[split_size:]][:]
train_y, val_y = train_y[shuffle[:split_size]][:], train_y[shuffle[split_size:]][:]

train_y =np.ravel(train_y)
val_y =np.ravel(val_y)

## Random Forest regression
print('\n')
print('Start regression')
regr = RandomForestRegressor(max_depth=10, random_state=0,verbose=1,n_estimators=10,n_jobs=-1)
regr.get_params()
#regr.set_params(n_estimators=10)
regr.fit(train_x, train_y)
print('\n')
print(regr.feature_importances_)

outTrain = regr.predict(train_x)
MeanErrorTrain = np.mean(abs(outTrain-train_y))
print('Mean training error random forest: {0}'.format(MeanErrorTrain))
print('\n')
outVal= regr.predict(val_x)
MeanErrorVal = np.mean(abs(outVal-val_y))
print('Mean validation error random forest: {0}'.format(MeanErrorVal))
print('\n')

#fig1=plt.figure()
#plt.plot(outTrain,'-r')
#plt.plot(train_y,'-b')
#
#    
#fig2=plt.figure()
#plt.plot(outVal,'-r')
#plt.plot(val_y,'-b')
#
#fig3=plt.figure()
#plt.plot(abs(outVal-val_y),'-r')



### Evaluate SVM
clf = svm.NuSVR(kernel='rbf')
clf.fit(train_x, train_y)

outTrainSVM = clf.predict(train_x)

MeanErrorTrainSVM = np.mean(abs(outTrainSVM-train_y))
print('Mean training error SVM: {0}'.format(MeanErrorTrainSVM))
print('\n')
outValSVM= clf.predict(val_x)
MeanErrorValSVM = np.mean(abs(outValSVM-val_y))
print('Mean validation error SVM: {0}'.format(MeanErrorValSVM))
print('\n')


### Evaluate linear model
#reg = linear_model.Lasso(alpha = 0.1)  #(36/22%)
#reg = linear_model.BayesianRidge()   #(22/22%)
reg = linear_model.Ridge (alpha = .5)   #(22/21%)
#reg = linear_model.LinearRegression()  #(20/20%)
reg.fit(train_x, train_y)

outTrainlinear = reg.predict(train_x)

MeanErrorTrainlinear= np.mean(abs(outTrainlinear-train_y))
print('Mean training error linear: {0}'.format(MeanErrorTrainlinear))
print('\n')
outVallinear= clf.predict(val_x)
MeanErrorVallinear = np.mean(abs(outVallinear-val_y))
print('Mean validation error linear: {0}'.format(MeanErrorVallinear))
print('\n')


##
a=regr.feature_importances_
print('\n')
for i in range(len(a)):
    print('{0} | {1:0.2f}%'.format(ExtractObj.FeatureTrainingDict['Name'][i],a[i]*100))
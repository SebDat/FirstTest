import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from JobDataExtraction import JobDataExtraction
from os import listdir, walk, getcwd
from os.path import isfile, join
import pickle
from datetime import datetime


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

mode='NN_2layers'
if(mode=='linear'):
    #allows to take the bias of the linear model into account
    nParam += 1   
    train_x = np.ones((FeatureMat.shape[0],FeatureMat.shape[1]+1))
    train_x[:,0:FeatureMat.shape[1]] = FeatureMat
elif(mode=='NN_1layer' or mode=='NN_2layers'):
    train_x = FeatureMat
else:
    print('Select avalable mode')     

train_y = OutputVect

### Separate training and validationb data
#Shuffle the exemple vector
shuffle = np.linspace(0,train_x.shape[0]-1,train_x.shape[0],dtype=int)

shuffle = ArrayShuffle(shuffle)
    
split_size = int(train_x.shape[0]*0.90)

train_x, val_x = train_x[shuffle[:split_size]][:], train_x[shuffle[split_size:]][:]
train_y, val_y = train_y[shuffle[:split_size]][:], train_y[shuffle[split_size:]][:]


#Opt info
batch_size = 1000
learning_rate = 0.001	#initially 0.05

device_name = "/gpu:0" #"/gpu:0"  #"/cpu:0"

with tf.device(device_name):
    
    ## NN architecture
    tf.reset_default_graph()
################################################################################################
#### Feedforward linear model
################################################################################################
    if(mode=='linear'):
        inputs_train = tf.placeholder(tf.float32, [batch_size,nParam])
        output_train = tf.placeholder(tf.float32, [batch_size,1])
        
        inputs_test = tf.placeholder(tf.float32, [train_x.shape[0],nParam])
        output_test = tf.placeholder(tf.float32, [train_y.shape[0],1])
        
        inputs_valid = tf.placeholder(tf.float32, [val_x.shape[0],nParam])
        output_valid = tf.placeholder(tf.float32, [val_y.shape[0],1])
        
        W = tf.Variable(tf.random_uniform([nParam,1],0,0.01))
        output_layer = tf.matmul(inputs_train,W)
        output_test = tf.matmul(inputs_test,W)
        output_valid = tf.matmul(inputs_valid,W)

################################################################################################

################################################################################################
### Feedforward 1 hidden layer NN (relu)
################################################################################################
    if(mode=='NN_1layer'):
        # number of neurons in each layer
        input_num_units = nParam
        hidden_num_units = 10
        output_num_units = 1
        
        inputs_train = tf.placeholder(tf.float32, [batch_size,nParam])
        output_train = tf.placeholder(tf.float32, [batch_size,1])
        
        inputs_test = tf.placeholder(tf.float32, [train_x.shape[0],nParam])
        output_test = tf.placeholder(tf.float32, [train_y.shape[0],1])
        
        weights = {
            'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
            'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
        }
        
        biases = {
            'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
            'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
        }
        
        hidden_layer = tf.add(tf.matmul(inputs_train, weights['hidden']), biases['hidden'])
        hidden_layer = tf.nn.relu(hidden_layer)
        
        output_layer = tf.add(tf.matmul(hidden_layer, weights['output']), biases['output'])  #sigmoid to have results contrained to [0;1]
    
        hidden_layer_test = tf.add(tf.matmul(inputs_test, weights['hidden']), biases['hidden'])
        hidden_layer_test = tf.nn.relu(hidden_layer_test)
        
        output_test = tf.add(tf.matmul(hidden_layer_test, weights['output']), biases['output'])

################################################################################################


################################################################################################
### Feedforward 2 hidden layer NN (relu)
################################################################################################
    if(mode=='NN_2layers'):
        # number of neurons in each layer
        input_num_units = nParam
        hidden_num_units1 = 5
        hidden_num_units2 = 5
        output_num_units = 1
        
        inputs_train = tf.placeholder(tf.float32, [batch_size,nParam])
        output_train = tf.placeholder(tf.float32, [batch_size,1])
        
        inputs_test = tf.placeholder(tf.float32, [train_x.shape[0],nParam])
        output_test = tf.placeholder(tf.float32, [train_y.shape[0],1])
        
        weights = {
            'hidden1': tf.Variable(tf.random_normal([input_num_units, hidden_num_units1], seed=seed)),
            'hidden2': tf.Variable(tf.random_normal([hidden_num_units1, hidden_num_units2], seed=seed)),
            'output': tf.Variable(tf.random_normal([hidden_num_units2, output_num_units], seed=seed))
        }
        
        biases = {
            'hidden1': tf.Variable(tf.random_normal([hidden_num_units1], seed=seed)),
            'hidden2': tf.Variable(tf.random_normal([hidden_num_units2], seed=seed)),        
            'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
        }
        
        hidden_layer1 = tf.add(tf.matmul(inputs_train, weights['hidden1']), biases['hidden1'])
        hidden_layer1 = tf.nn.relu(hidden_layer1)
        
        hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['hidden2']), biases['hidden2'])
        hidden_layer2 = tf.nn.relu(hidden_layer2)
        
        output_layer = tf.nn.relu(tf.add(tf.matmul(hidden_layer2, weights['output']), biases['output']))  #sigmoid to have results contrained to [0;1]
    
        hidden_layer_test1 = tf.add(tf.matmul(inputs_test, weights['hidden1']), biases['hidden1'])
        hidden_layer_test1 = tf.nn.relu(hidden_layer_test1)
    
        hidden_layer_test2 = tf.add(tf.matmul(hidden_layer_test1, weights['hidden2']), biases['hidden2'])
        hidden_layer_test2 = tf.nn.relu(hidden_layer_test2)
        
        output_test = tf.nn.relu(tf.add(tf.matmul(hidden_layer_test2, weights['output']), biases['output']))

################################################################################################

    #Below we obtain the loss by taking the sum of squares difference between the target and prediction values.
    loss = tf.reduce_sum(tf.square(output_train - output_layer))
    #trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    #trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    trainer = tf.train.AdamOptimizer().minimize(loss)
    
    
    ## Training the network
    init = tf.initialize_all_variables()

num_episodes = 50000

startTime = datetime.now()


err=[]
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    
    for i in range(num_episodes):
        avg_cost = 0
        total_batch = int(train_x.shape[0]/batch_size)        
        for j in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train', nParam, train_x, train_y)
            _, c = sess.run([trainer, loss], feed_dict = {inputs_train: batch_x, output_train: batch_y})
            avg_cost += c / total_batch
        if(i%10 ==0):    
            print("Epoch:", (i+1), "cost =", "{:.5f}".format(100*avg_cost/batch_size))
        err.append(avg_cost)
    print("\nTraining complete!")
    
    out=sess.run(output_test, feed_dict={inputs_test: train_x})
    ErrTrain = sess.run(tf.sqrt(tf.reduce_sum(tf.square(out-train_y))))
    print('Training Accuracy Linear Model: ' ,ErrTrain) 
    
    
    print("\n" * 2)
    print("Device:", device_name)
    print("Time taken:", datetime.now() - startTime)

    print("\n" * 2)
    
    a=np.where(out>1)
    
    out2 = np.array(out)
    out2[a[0]] = 1
    
    a=np.where(out<0)
    out2[a[0]] = 0
    
    fig1=plt.figure()
    plt.plot(out,'-r')
    plt.plot(train_y,'-b')
    plt.plot(out2,'-g')
    
    fig2=plt.figure()
    plt.plot(abs(out-train_y),'-r')
    
    fig3=plt.figure()
    plt.plot(abs(out2-train_y),'-r')
    
    print('Mean abs training error {0:.2f}%'.format(np.mean(abs(out-train_y))))
    print('Mean abs training error {0:.2f}%'.format(np.mean(abs(out2-train_y))))
    print('Learning Rate: {0}'.format(learning_rate))

#    out_test=sess.run(output_valid, feed_dict={inputs_valid: val_x}) 
#    ErrTest = sess.run(tf.sqrt(tf.reduce_sum(tf.square(out_test-val_y))))
#    print('Test Accuracy Linear Model: ' ,ErrTest)





fig4=plt.figure()
x1=np.linspace(1,len(err),len(err))
plt.plot(x1,100*np.array(err)/batch_size)  # on utilise la fonction sinus de Numpy
plt.ylabel('Error')
plt.xlabel("Epoch")
plt.ylim((0,50))
plt.show()


import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt #to vizualize output

model_path = "/home/katou/Python/BTC trade code/model/model_v070717.ckpt"

seed = 128  
rng = np.random.RandomState(seed)
root_dir = os.path.abspath('../..')   #katou home
data_dir = os.path.join(root_dir, 'Python','BTC trade code','data')
sub_dir = os.path.join(root_dir, 'Python','BTC trade code','sub')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)

#data preparation routine

n_days=7   #number of past days to take into account to make prediction #initally 7

features=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  #features to take into account Opening price/Volume BTC (very useful)/BCHAIN-ATRCT(not super useful)/BCHAIN-ETRAV(not super useful)/BCHAIN-ETRVU(not super useful)/BCHAIN-MKTCP(not super useful)/BCHAIN-TRVOU(not super useful)



#col_feat=[2, 6, 9, 10, 11, 10, 13] #column of the corresponding feature in the .csv file

#temp = []
#for ft in range(len(features)):
#	#print(ft)
#	if features[ft]==1:
#		temp.append(col_feat[ft])
#
#col_feat=temp


#raw = pd.read_csv(os.path.join(data_dir, 'raw_data.csv'))
raw = pd.read_csv('/home/katou/Python/BTC trade code/blockchain_data_06-07-17_raw.csv')
#raw = pd.read_csv('/home/katou/Python/BTC trade code/BCHARTS-KRAKENEUR-corr+volume.csv')  #test the corrected data

#raw.Open will output the array of the column under Open 



bchain_price=raw.mark_price/raw.mark_price.max()
bchain_volBTC=raw.Volume_BTC/raw.Volume_BTC.max()
bchain_time=raw.BCHAIN_ATRCT/raw.BCHAIN_ATRCT.max()	
bchain_tradBTC=raw.BCHAIN_ETRAV/raw.BCHAIN_ETRAV.max()
bchain_tradEUR=raw.BCHAIN_ETRVU/raw.BCHAIN_ETRVU.max()	
bchain_marketcap=raw.BCHAIN_MKTCP/raw.BCHAIN_MKTCP.max()
bchain_exchVol=raw.BCHAIN_TRVOU/raw.BCHAIN_TRVOU.max()	
bchain_cost_trans=raw.cost_trans/raw.cost_trans.max()
bchain_mark_cap=raw.mark_cap/raw.mark_cap.max()
bchain_conf_time=raw.conf_time/raw.conf_time.max()
bchain_miner_rev=raw.miner_rev/raw.miner_rev.max()
bchain_N_trans=raw.N_trans/raw.N_trans.max()
bchain_out_vol=raw.out_vol/raw.out_vol.max()
bchain_trade_vol=raw.trade_vol/raw.trade_vol.max()
bchain_trade_fee=raw.trade_fee/raw.trade_fee.max()


#x=np.linspace(1,len(bchain_price),len(bchain_price))
#plt.plot(x,bchain_price)  # on utilise la fonction sinus de Numpy
#plt.ylabel('fonction sinus')
#plt.xlabel("l'axe des abcisses")
#plt.show()

########
#REMOVE DATA FROM 481 to 526, data corrupted

#i1=np.linspace(0,480,481)
#i2=np.linspace(527+7,len(bchain_price),len(bchain_price)-527-7+1)

#mask_1=np.concatenate([i1,i2])

#bchain_price=bchain_price[mask_1]
#bchain_volBTC=bchain_volBTC[mask_1]
#bchain_time=bchain_time[mask_1]
#bchain_tradBTC=bchain_tradBTC[mask_1]
#bchain_tradEUR=bchain_tradEUR[mask_1]
#bchain_marketcap=bchain_marketcap[mask_1]	
#bchain_exchVol=bchain_exchVol[mask_1]



#construct array of selected features

temp = []
for ft in range(len(features)):
	#print(ft)
	if features[ft]==1:
		if ft==0:
			temp.append(bchain_price)
		elif ft==1:
			temp.append(bchain_volBTC)
		elif ft==2:
			temp.append(bchain_time)
		elif ft==3:
			temp.append(bchain_tradBTC)
		elif ft==4:
			temp.append(bchain_tradEUR)
		elif ft==5:
			temp.append(bchain_marketcap)
		elif ft==6:
			temp.append(bchain_exchVol)
		elif ft==7:
			temp.append(bchain_cost_trans)
		elif ft==8:
			temp.append(bchain_mark_cap)
		elif ft==9:
			temp.append(bchain_conf_time)
		elif ft==10:
			temp.append(bchain_miner_rev)
		elif ft==11:
			temp.append(bchain_N_trans)
		elif ft==12:
			temp.append(bchain_out_vol)
		elif ft==13:
			temp.append(bchain_trade_vol)
		elif ft==14:
			temp.append(bchain_trade_fee)

#col_feat=temp
col_feat=np.stack(temp) #to get an numpy array

#assemble features for each example of the dataset

n_example=col_feat.shape[1]-n_days


#prediction at n+k day

kk=1

temp=[]
next_p=[]
for i in range(n_example+1-kk):
	l=[]
	for ii in range(len(col_feat)):
		l.append(col_feat[ii][i+kk:i+kk+n_days])
	temp.append(l) 
	next_p.append(100*(bchain_price[i]-bchain_price[i+kk])/(bchain_price[i+kk]+1e-10))

#FEAT=temp
FEAT=np.stack(temp) #to get an numpy array

FEAT=FEAT.reshape(FEAT.shape[0],FEAT.shape[1]*FEAT.shape[2])

next_p = np.stack(next_p)

#normally the FEAT array should have a dimension of n_example*n_days*num_feature, each feature vector should be n_days*num_feature  (7*7 for instance)


#x=np.linspace(1,len(next_p),len(next_p))
#plt.plot(x,next_p)  # on utilise la fonction sinus de Numpy
#plt.ylabel('fonction sinus')
#plt.xlabel("l'axe des abcisses")
#plt.show()


#create label, we can define the % of variation of the price at day n+1 as the output which is then put into classes +- 1% +- 1-2% +- 2-5% +- 5-10% +- 10-inf% if these classes are too large we can redo them

y=[]

for i in range(len(next_p)):
	if next_p[i] > 10:
		y.append(0)
	elif next_p[i] > 5 and next_p[i] <= 10:
		y.append(1)
	elif next_p[i] > 2 and next_p[i] <= 5:
		y.append(2)
	elif next_p[i] > 1 and next_p[i] <= 2:
		y.append(3)
	elif next_p[i] >= 0 and next_p[i] <= 1:
		y.append(4)
	elif next_p[i] < 0 and next_p[i] >= -1:
		y.append(5)
	elif next_p[i] < -1 and next_p[i] >= -2:
		y.append(6)
	elif next_p[i] < -2 and next_p[i] >= -5:
		y.append(7)
	elif next_p[i] < -5 and next_p[i] >= -10:
		y.append(8)
	elif next_p[i] < -10:
		y.append(9)

y=np.stack(y)

#x=np.linspace(1,len(y),len(y))
#plt.plot(x,y)  # on utilise la fonction sinus de Numpy
#plt.ylabel('fonction sinus')
#plt.xlabel("l'axe des abcisses")
#plt.show()

#group of functions pre-defined
def dense_to_one_hot(labels_dense, num_classes=10):
    #"""Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot

def preproc(unclean_batch_x):
    #"""Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

#def batch_creator(batch_size, dataset_length, dataset_name):
#    """Create batch with random samples and return appropriate format"""
#    batch_mask = rng.choice(dataset_length, batch_size)
#    
#    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)
#    batch_x = preproc(batch_x)
#    
#   if dataset_name == 'train':
#        batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
#       batch_y = dense_to_one_hot(batch_y)
#        
#    return batch_x, batch_y


def batch_creator(batch_size, dataset_length, dataset_name):
    #"""Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)
    #batch_x = preproc(batch_x)
    
    if dataset_name == 'train':
        batch_y = train_y[batch_mask] 
    return batch_x, batch_y

#use function dense_to_one_hot to convert classes
y_label=dense_to_one_hot(y, num_classes=10)


#separate data into training and testing - DO IT IN A SECOND TIME

#########
###Removal of corrupted data
#########

#i1=np.linspace(0,480,481)
#i2=np.linspace(527+n_days,len(next_p)-1,len(next_p)-527-n_days)

#mask1=np.concatenate([i1,i2])
#mask1=mask1.astype(int)
#y_label=y_label[mask1]
#next_p=next_p[mask1]
#FEAT=FEAT[mask1]

####split data to get train an validation set

## test model on recent data
#raw2 = pd.read_csv('/home/katou/Python/BTC trade code/results/testing-recent/BCHARTS-KRAKENEUR_recent.csv')
#raw2 = pd.read_csv('/home/katou/Python/BTC trade code/results/testing-recent/BCHARTS-KRAKENEUR_recent_corr.csv')

#raw.Open will output the array of the column under Open 


####change the max by the previous max !!!!!!!
#bchain_price=raw2.Open/raw.Open.max()
#bchain_volBTC=raw2.Volume_BTC/raw.Volume_BTC.max()
#bchain_time=raw2.BCHAIN_ATRCT/raw.BCHAIN_ATRCT.max()	
#bchain_tradBTC=raw2.BCHAIN_ETRAV/raw.BCHAIN_ETRAV.max()
#bchain_tradEUR=raw2.BCHAIN_ETRVU/raw.BCHAIN_ETRVU.max()	
#bchain_marketcap=raw2.BCHAIN_MKTCP/raw.BCHAIN_MKTCP.max()	
#bchain_exchVol=raw2.BCHAIN_TRVOU/raw.BCHAIN_TRVOU.max()


##########################################################################"
###WRITE ROUTINE FOR THE DATA PREPARATION
#############################################################################



#temp = []
#for ft in range(len(features)):
#	#print(ft)
#	if features[ft]==1:
#		if ft==0:
#			temp.append(bchain_price)
#		elif ft==1:
#			temp.append(bchain_volBTC)
#		elif ft==2:
#			temp.append(bchain_time)
#		elif ft==3:
#			temp.append(bchain_tradBTC)
#		elif ft==4:
#			temp.append(bchain_tradEUR)
#		elif ft==5:
#			temp.append(bchain_marketcap)
#		elif ft==6:
#			temp.append(bchain_exchVol)
#
#col_feat=temp
#col_feat=np.stack(temp) #to get an numpy array

#assemble features for each example of the dataset

#n_example=col_feat.shape[1]-n_days


#temp=[]
#next_p_test=[]
#for i in range(n_example):
#	l=[]
#	for ii in range(len(col_feat)):
#		l.append(col_feat[ii][i+1:i+1+n_days])
#	temp.append(l) 
#	next_p_test.append(100*(bchain_price[i]-bchain_price[i+1])/(bchain_price[i+1]+1e-10))
#
#FEAT=temp
#FEAT_test=np.stack(temp) #to get an numpy array

#FEAT_test=FEAT_test.reshape(FEAT_test.shape[0],FEAT_test.shape[1]*FEAT_test.shape[2])

#next_p_test = np.stack(next_p_test)



#normally the FEAT array should have a dimension of n_example*n_days*num_feature, each feature vector should be n_days*num_feature  (7*7 for instance)


#x=np.linspace(1,len(next_p),len(next_p))
#plt.plot(x,next_p)  # on utilise la fonction sinus de Numpy
#plt.ylabel('fonction sinus')
#plt.xlabel("l'axe des abcisses")
#plt.show()


#create label, we can define the % of variation of the price at day n+1 as the output which is then put into classes +- 1% +- 1-2% +- 2-5% +- 5-10% +- 10-inf% if these classes are too large we can redo them

#y_test=[]

#for i in range(len(next_p_test)):
#	if next_p_test[i] > 10:
#		y_test.append(0)
#	elif next_p_test[i] > 5 and next_p_test[i] <= 10:
#		y_test.append(1)
#	elif next_p_test[i] > 2 and next_p_test[i] <= 5:
#		y_test.append(2)
#	elif next_p_test[i] > 1 and next_p_test[i] <= 2:
#		y_test.append(3)
#	elif next_p_test[i] >= 0 and next_p_test[i] <= 1:
#		y_test.append(4)
#	elif next_p_test[i] < 0 and next_p_test[i] >= -1:
#		y_test.append(5)
#	elif next_p_test[i] < -1 and next_p_test[i] >= -2:
#		y_test.append(6)
#	elif next_p_test[i] < -2 and next_p_test[i] >= -5:
#		y_test.append(7)
#	elif next_p_test[i] < -5 and next_p_test[i] >= -10:
#		y_test.append(8)
#	elif next_p_test[i] < -10:
#		y_test.append(9)

#y_test=np.stack(y_test)
#y_label_test=dense_to_one_hot(y_test, num_classes=10)


#test_x=FEAT_test


###########################################################################
##### DEFINE NEURAL NETWORK
### set all variables

# number of neurons in each layer
input_num_units = n_days*col_feat.shape[0]
hidden_num_units = 500  #initally 500
output_num_units = 10

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])


# set remaining variables
epochs = 300		#initally 5
batch_size = 32		#initally 128
learning_rate = 0.005	#initially 0.05

### define weights and biases of the neural network 

weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

#obsolete
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#obsolete
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

#JUST FOR TESTING
train_x=FEAT

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()




split_size = int(train_x.shape[0]*0.70)

train_x, val_x = train_x[:split_size][:], train_x[split_size:][:]
train_y, val_y = y_label[:split_size][:], y_label[split_size:][:]

err=[]
with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train_x.shape[0]/batch_size)
	#total_batch = int(train.shape[0]/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
        err.append(avg_cost)
    print("\nTraining complete!")
    out=sess.run(output_layer, feed_dict={x: train_x})
    pred_temp = sess.run(tf.equal(tf.argmax(out, 1), tf.argmax(train_y, 1)))
    out_lab = sess.run(tf.argmax(out, 1))
    y_lab = sess.run(tf.argmax(train_y, 1))
    accuracy = sess.run(tf.reduce_mean(tf.cast(pred_temp, "float")))   
    print('Training Accuracy: ' ,accuracy) 
    # Save model weights to disk 
    save_path = saver.save(sess, model_path) 
    print("Model saved in file: %s" % save_path) 
    # Use model on new data
    out_test=sess.run(output_layer, feed_dict={x: val_x}) 
    pred_temp_test = sess.run(tf.equal(tf.argmax(out_test, 1), tf.argmax(val_y, 1))) 
    out_lab_test = sess.run(tf.argmax(out_test, 1))
    y_lab_test = sess.run(tf.argmax(val_y,1)) 
    accuracy_test = sess.run(tf.reduce_mean(tf.cast(pred_temp_test, "float"))) 
    delta=sess.run(tf.argmax(out_test, 1)-tf.argmax(val_y, 1))
    print('Test Accuracy: ' ,accuracy_test)

#


D=np.mean(abs(delta))
print(D)

#plot training evolution

#err=np.cast(err)

x1=np.linspace(1,len(err),len(err))
plt.plot(x1,err)  # on utilise la fonction sinus de Numpy
plt.ylabel('fonction sinus')
plt.xlabel("l'axe des abcisses")
plt.show()

x1=np.linspace(1,len(out_lab),len(out_lab))
plt.plot(x1,out_lab,x1,y_lab)  # on utilise la fonction sinus de Numpy
plt.ylabel('fonction sinus')
plt.xlabel("l'axe des abcisses")
plt.show()

x1=np.linspace(1,len(out_lab_test),len(out_lab_test))
plt.plot(x1,out_lab_test,x1,y_lab_test)  # on utilise la fonction sinus de Numpy
plt.ylabel('fonction sinus')
plt.xlabel("l'axe des abcisses")
plt.show()


import gym
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


env = gym.make('NChain-v0')

model = Sequential()
model.add(InputLayer(batch_input_shape=(1, 5)))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#In this example the Loss = (r+gamma*max_a'_Q(s',a') - Q(s,a))^2
#basically, we try to predict the discounted Q value with the NN
#Note that we are not using dueling Q network so the learning might be noisy
# with a secondary network updated slowly, the learning should be more stable

r_avg_list = []

# now execute the q learning
y = 0.95
eps = 0.05
decay_factor = 0.999

num_episodes = 10000
for i in range(num_episodes):
    s = env.reset()
    eps *= decay_factor
    #if i % 100 == 0:
    print("Episode {} of {}".format(i + 1, num_episodes))
    done = False
    r_sum = 0
    #c=0
    tstart = datetime.now()
    while not done:
        #c+=1
        #print('Game: {0} - Action number: {1}'.format(i + 1,c))
        if np.random.random() < eps:
            a = np.random.randint(0, 2)
        else:
            a = np.argmax(model.predict(np.identity(5)[s:s + 1]))

        new_s, r, done, _ = env.step(a)
        

        target = r + y * np.max(model.predict(np.identity(5)[new_s:new_s + 1]))

        target_vec = model.predict(np.identity(5)[s:s + 1])[0]
        target_vec[a] = target

        model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
        s = new_s
        r_sum += r
    tend = datetime.now()
    print('Game took: {0} Current Reward: {1}'.format(tend-tstart,r_sum / 1000))    
    r_avg_list.append(r_sum / 1000)
    
plt.plot(r_avg_list,'-bo')
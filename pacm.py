import numpy as np
import gym
import tensorflow  as tf
from  tensorflow.contrib.layers import flatten, conv2d ,fully_connected
from collections import deque, Counter
import random
from datetime import datetime

def preprocess_observation(obs):
	#crop and resize
	img = obs[1:176:2, ::2]
	#greyscale
	img = img.mean(axis=2)
	color = np.array([210,164,74]).mean()
	img[img==color] = 0

	#normalize image -1 to 1
	img = (img - 128) / 128 - 1
	#one channel 88x88
	return img.reshape(88,80,1)

#define graph for q-network
def q_network(X,name_scope):
	#Initialize layers
	initializer_variables = tf.contrib.layers.variance_scaling_initializer()
	
	with tf.variable_scope(name_scope) as scope:
		#conv1
		layer_1 = conv2d(X,num_outputs=32, kernel_size=(8,8),stride=4,padding='SAME',weights_initializer=initializer_variables)
		tf.summary.histogram('layer_1',layer_1)
		
		layer_2 = conv2d(layer_1,num_outputs=64,kernel_size=(4,4),stride=2,padding='SAME',weights_initializer=initializer_variables)
		tf.summary.histogram('layer_2',layer_2)

		layer_3 = conv2d(layer_2,num_outputs=64,kernel_size=(3,3),stride=1,padding='SAME',weights_initializer=initializer_variables)
		tf.summary.histogram('layer_3',layer_3)

		flat = flatten(layer_3)
		
		fc = fully_connected(flat,num_outputs=128,weights_initializer=initializer_variables)
		tf.summary.histogram('fc',fc)

		output = fully_connected(fc,num_outputs=num_outputs,activation_fn=None,weights_initializer=initializer_variables)
		tf.summary.histogram('output',output)

		vars = {v.name[len(scope.name):]: v for v in tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope.name) }
		return vars,output

def epsilon_greedy(action,step):
	p = np.random.random(1).squeeze()
	epsilon = max(eps_min,eps_max - (eps_max - eps_min) * step/eps_decay_steps)
	if np.random.rand() < epsilon:
		return np.random.randint(num_outputs)
	else:
		return action

def sample_memories(batch_size):
	perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
	mem = np.array(exp_buffer)[perm_batch]
	return mem[:,0], mem[:,1],mem[:,2],mem[:,3],mem[:,4]


		

env = gym.make("MsPacman-v0")
num_outputs = env.action_space.n
print("action space: "+str(num_outputs))
epsilon = 0.5
eps_min = 0.05
eps_max = 0.8
eps_decay_steps = 50000

buffer_len = 20000
exp_buffer = deque(maxlen=buffer_len)

num_episodes = 80
batch_size = 68
input_shape = (None,88,88,1)
learning_rate = 0.001
X_shape = (None,88,80,1)
discount_factor = 0.87

global_step = 0
copy_steps = 100
steps_train = 4
start_steps = 2000

logdir = 'logs'

tf.reset_default_graph()

#placeholder for the input
X = tf.placeholder(tf.float32,shape=X_shape)

#we defina a boolean to toggle the training
in_training_mode = tf.placeholder(tf.bool)

#main q-network
mainQ, mainQ_output = q_network(X,'mainQ')

#target q-network
targetQ, targetQ_outputs = q_network(X,'targetQ')

#define placeholder for action values
X_action = tf.placeholder(tf.int32,shape=(None,))
Q_action = tf.reduce_sum(targetQ_outputs * tf.one_hot(X_action,num_outputs),axis=-1,keepdims=True)

#copy the main q-network to the target q-network
copy_op = [tf.assign(main_name,targetQ[var_name]) for var_name,main_name in mainQ.items()]
copy_target_to_main = tf.group(*copy_op)

#define placeholder for our output
y = tf.placeholder(tf.float32,shape=(None,1))

# now we calculate the loss wich is the difference between actual value and predicted value
loss = tf.reduce_mean(tf.square(y - Q_action))

#adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

loss_summary = tf.summary.scalar('Loss',loss)
merge_summary = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())

saver = tf.train.Saver()


with tf.Session() as sess:
	init.run()

	# for each episode
	for i in range(num_episodes):
		done = False
		obs = env.reset()
		epoch = 0
		episodic_reward = 0
		actions_counter = Counter()
		episodic_loss = []

		while not done:
			#process the game screen
			obs = preprocess_observation(obs)

			#feed the game screen and get the Q values for each action
			actions = mainQ_output.eval(feed_dict={X:[obs],in_training_mode:False})

			#get the action
			action = np.argmax(actions, axis=-1)
			actions_counter[str(action)] += 1

			action = epsilon_greedy(action,global_step)

			next_obs,reward,done,_ = env.step(action)
			
			#store experience replay buffer
			exp_buffer.append([obs,action,preprocess_observation(next_obs),reward,done])
			#solo despues de n steps entrenamos la q-network con data del replay buffer
			if global_step % steps_train == 0 and global_step > start_steps:
				#sample experience
				o_obs,o_act,o_next_obs,o_rew,o_done = sample_memories(batch_size)

				o_obs = [x for x in o_obs]
				o_next_obs = [x for x in o_next_obs]
			
				next_act = mainQ_output.eval(feed_dict={X:o_next_obs,in_training_mode:False})

				o_2_obs,o_2_act,o_2_next_obs,o_2_rew,o_2_done = sample_memories(batch_size)

				o_2_obs = [x for x in o_2_obs]
				o_2_next_obs = [x for x in o_2_next_obs]

				next_2_act = mainQ_output.eval(feed_dict={X:o_2_next_obs,in_training_mode:False})

				discount = discount_factor * np.max(next_act,axis=-1) + (discount_factor**2)*np.max(next_2_act,axis=-1) 

				y_batch = ((o_rew + o_2_rew) + (discount)) * (1-o_done)*(1-o_2_done)

				train_loss, _ = sess.run([loss,training_op],feed_dict={X:o_obs, y:np.expand_dims(y_batch,axis=-1),X_action:o_act, in_training_mode:True})
				episodic_loss.append(train_loss)
			#copy main q-net weights to target q-network
			if (global_step+1) % copy_steps == 0 and global_step > start_steps:
				copy_target_to_main.run()
			obs = next_obs
			epoch += 1
			global_step += 1
			episodic_reward += reward 
		print('Epoch',epoch,'Reward',reward,episodic_reward,)
	if ( i % 10 == 0):
		save_path = saver.save(sess,"/home/ubuntu/openai/pacman/model/model2.ckpt")
		print("model saved in path: %s" % save_path)












								














		























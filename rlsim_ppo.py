# import gym
from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
import random
import numpy as np
import scipy.signal
import rlsim_env
import tensorflow as tf
from collections import deque
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

seed = 0
np.random.seed(seed)

class Policy(object):
    def __init__(self, obs_dim, act_dim, clip_range=0.2,
                 epochs=10, lr=3e-5, convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], hiddens=[256], max_std=0.1,
                 seed=0):
        
        self.seed=0
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.clip_range = clip_range
        
        self.epochs = epochs
        self.lr = lr
        self.convs = convs
        self.hiddens = hiddens
        self.max_std = max_std
        
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()
            self.variables = tf.global_variables()
            
    def _placeholders(self):
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, [None] + list(self.obs_dim), 'obs')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')

        # learning rate:
        self.lr_ph = tf.placeholder(tf.float32, (), 'lr')
        
        # place holder for old parameters
        self.old_std_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_std')
        self.old_mean_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

    def _policy_nn(self):
        
#         hid1_size = self.hdim
#         hid2_size = self.hdim
        out = self.obs_ph
        # CONV LAYERS
        for num_outputs, kernel_size, stride in self.convs:
            out = tf.contrib.layers.convolution2d(out,
                                                  num_outputs=num_outputs,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  activation_fn=tf.nn.relu)
        out = tf.layers.flatten(out)    
        
        # HIDDEN LAYERS
        for hidden in self.hiddens:
            out = tf.layers.dense(out, hidden, tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed))
        
#         out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
#                               kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="h1")
#         out = tf.layers.dense(out, hid2_size, tf.tanh,
#                               kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed), name="h2")
        
        # MEAN FUNCTION
        self.mean = tf.layers.dense(out, self.act_dim, tf.sigmoid,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed),
                                    name="mean")
        # UNIT VARIANCE
        self.logits_std = tf.get_variable("logits_std",shape=(1,),initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed))
        self.std = self.max_std*tf.ones_like(self.mean)*tf.sigmoid(self.logits_std)
        
        # SAMPLE OPERATION
        self.sample_action = self.mean + tf.random_normal(tf.shape(self.mean),seed= self.seed)*self.std
        
    def _logprob(self):
        # PROBABILITY WITH TRAINING PARAMETER
        y = self.act_ph 
        mu = self.mean
        sigma = self.std
        
        self.logp = tf.reduce_sum(-0.5*tf.square((y-mu)/sigma)-tf.log(sigma)- 0.5*np.log(2.*np.pi),axis=1)

        # PROBABILITY WITH OLD (PREVIOUS) PARAMETER
        old_mu_ph = self.old_mean_ph
        old_sigma_ph = self.old_std_ph
                
        self.logp_old = tf.reduce_sum(-0.5*tf.square((y-old_mu_ph)/old_sigma_ph)-tf.log(old_sigma_ph)- 0.5*np.log(2.*np.pi),axis=1)
        
    def _kl_entropy(self):

        mean, std = self.mean, self.std
        old_mean, old_std = self.old_mean_ph, self.old_std_ph
 
        log_std_old = tf.log(old_std)
        log_std_new = tf.log(std)
        frac_std_old_new = old_std/std

        # KL DIVERGENCE BETWEEN TWO GAUSSIAN
        kl = tf.reduce_sum(log_std_new - log_std_old + 0.5*tf.square(frac_std_old_new) + 0.5*tf.square((mean - old_mean)/std)- 0.5,axis=1)
        self.kl = tf.reduce_mean(kl)
        
        # ENTROPY OF GAUSSIAN
        entropy = tf.reduce_sum(log_std_new + 0.5 + 0.5*np.log(2*np.pi),axis=1)
        self.entropy = tf.reduce_mean(entropy)
        
    def _loss_train_op(self):
        
        # Proximal Policy Optimization CLIPPED LOSS FUNCTION
        ratio = tf.exp(self.logp - self.logp_old) 
        clipped_ratio = tf.clip_by_value(ratio,clip_value_min=1-self.clip_range,clip_value_max=1+self.clip_range) 
        self.loss = -tf.reduce_mean(tf.minimum(self.advantages_ph*ratio,self.advantages_ph*clipped_ratio))
        
        # OPTIMIZER 
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs): # SAMPLE FROM POLICY
        feed_dict = {self.obs_ph: obs}
        sampled_action = self.sess.run(self.sample_action,feed_dict=feed_dict)
        return sampled_action
    
    def control(self, obs): # COMPUTE MEAN
        feed_dict = {self.obs_ph: obs}
        best_action = self.sess.run(self.mean,feed_dict=feed_dict)
        return best_action        
    
    def update(self, observes, actions, advantages, batch_size = 128): # TRAIN POLICY
        
        num_batches = max(observes.shape[0] // batch_size, 1)
        batch_size = observes.shape[0] // num_batches
        
        old_means_np, old_std_np = self.sess.run([self.mean, self.std],{self.obs_ph: observes}) # COMPUTE OLD PARAMTER
        for e in xrange(self.epochs):
            observes, actions, advantages, old_means_np, old_std_np = shuffle(observes, actions, advantages, old_means_np, old_std_np, random_state=self.seed)
            for j in xrange(num_batches): 
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: observes[start:end,:],
                     self.act_ph: actions[start:end,:],
                     self.advantages_ph: advantages[start:end],
                     self.old_std_ph: old_std_np[start:end,:],
                     self.old_mean_ph: old_means_np[start:end,:],
                     self.lr_ph: self.lr}        
                self.sess.run(self.train_op, feed_dict)
            
        feed_dict = {self.obs_ph: observes,
                 self.act_ph: actions,
                 self.advantages_ph: advantages,
                 self.old_std_ph: old_std_np,
                 self.old_mean_ph: old_means_np,
                 self.lr_ph: self.lr}             
        loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
        return loss, kl, entropy
    
    def close_sess(self):
        self.sess.close()


class Value(object):
    def __init__(self, obs_dim, epochs=20, lr=1e-4,  convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], hiddens=[256],  seed=0):
        self.seed = seed
    
        self.obs_dim = obs_dim
        self.epochs = epochs
        self.lr = lr
        self.convs = convs
        self.hiddens = hiddens
        
        self._build_graph()
        self._init_session()
        
    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, [None] + list(self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            
#             hid1_size = self.hdim 
#             hid2_size = self.hdim 

            out = self.obs_ph
            # CONV LAYERS
            for num_outputs, kernel_size, stride in self.convs:
                out = tf.contrib.layers.convolution2d(out,
                                                      num_outputs=num_outputs,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      activation_fn=tf.nn.relu)
            out = tf.layers.flatten(out)    
        
            # HIDDEN LAYERS
            for hidden in self.hiddens:
                out = tf.layers.dense(out, hidden, tf.tanh, kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed= self.seed))
            
#             out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
#                                   kernel_initializer=tf.random_normal_initializer(
#                                       stddev=0.01,seed=self.seed), name="h1")
#             out = tf.layers.dense(out, hid2_size, tf.tanh,
#                                   kernel_initializer=tf.random_normal_initializer(
#                                       stddev=0.01,seed=self.seed), name="h2")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=0.01,seed=self.seed), name='output')
            self.out = tf.squeeze(out)
            # self.out = out

            # L2 LOSS
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))
            
            # OPTIMIZER
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            
            self.init = tf.global_variables_initializer()
            self.variables = tf.global_variables()
    
    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        self.sess.run(self.init)

    def fit(self, x, y, batch_size=32):
        num_batches = max(x.shape[0] // batch_size, 1)
        x_train, y_train = x, y
        for e in xrange(self.epochs):
            x_train, y_train = shuffle(x_train, y_train, random_state=self.seed)
            for j in xrange(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                self.sess.run([self.train_op], feed_dict=feed_dict)
        feed_dict = {self.obs_ph: x_train,
                     self.val_ph: y_train}
        loss, = self.sess.run([self.loss], feed_dict=feed_dict)
        return loss

    def predict(self, x): # PREDICT VALUE OF THE GIVEN STATE
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)
        #return np.squeeze(y_hat)
        return y_hat

    def close_sess(self):
        self.sess.close()


def discount(x, gamma=0.99): # compute discount
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

def add_value(trajectories, val_func): # Add value estimation for each trajectories
    for trajectory in trajectories:
        observes = trajectory['observes']
        # print(np.shape(observes)) #debug
        values = val_func.predict(observes)
        trajectory['values'] = values

def add_gae(trajectories, gamma=0.99, lam=0.95): # generalized advantage estimation (for training stability)
    for trajectory in trajectories:
        rewards = trajectory['rewards']
        values = trajectory['values']
        
        # temporal differences
        # print(values) #debug
        if np.isscalar(values) == 0:
            tds = rewards - values + np.append(values[1:] * gamma, 0)
        else:
            tds = rewards - values
        advantages = discount(tds, gamma * lam)
        
        trajectory['advantages'] = advantages
        trajectory['returns'] = values+advantages

def build_train_set(trajectories):
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    returns = np.concatenate([t['returns'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])

    # Normalization of advantages 
    # In baselines, which is a github repo including implementation of PPO by OpenAI, 
    # all policy gradient methods use advantage normalization trick as belows.
    # The insight under this trick is that it tries to move policy parameter towards locally maximum point.
    # Sometimes, this trick doesnot work.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, returns

def run_episode(env, policy, animate=False, evaluation=False): # Run policy and collect (state, action, reward) pairs
    obs = env.reset()
    observes, actions, rewards, infos = [], [], [], []
    done = False
    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((-1,84,84,4))  # .reshape((1, -1))
        # print(obs[0][1][1][1]) #debug
        observes.append(obs)
        if evaluation:
            action = policy.control(obs).reshape((1, -1)).astype(np.float32)
        else:
            action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)

        # action_ = tf.squeeze(action)
        obs, reward, done, info = env.step(continuous_action(action[0]))
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        infos.append(info)
        
    return (np.concatenate(observes), np.concatenate(actions), np.array(rewards, dtype=np.float32), infos)

def run_policy(env, policy, episodes, evaluation=False): # collect trajectories. if 'evaluation' is ture, then only mean value of policy distribution is used without sampling.
    total_steps = 0
    trajectories = []
    for e in xrange(episodes):
        observes, actions, rewards, infos = run_episode(env, policy, evaluation=evaluation)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'infos': infos}
        trajectories.append(trajectory)
    return trajectories

def continuous_action(action):

    if len(action)==1:
        action = action[0]

    angle_cmd = 4*(action[0]-0.5)

    acc = action[1]

    pedal_cmd = acc
    #break_cmd = 2*(0.5-acc) if acc<0.5 else 0
    break_cmd = 0

    return [angle_cmd,pedal_cmd,break_cmd]


# env = gym.make('Pendulum-v0')
env = rlsim_env.make('straight_4lane_cam')

print 'updated'

seed = 0
np.random.seed(seed)
random.seed(seed)

obs_dim = env.observation_space.shape #.shape[0]
act_dim = 2 # env.action_space #.shape[0]

policy = Policy(obs_dim, act_dim, epochs=50, convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], hiddens=[256], lr=3e-4, clip_range=0.2,seed=seed)
val_func = Value(obs_dim, epochs=100, convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], hiddens=[256], lr=1e-3, seed=seed)

episode_size = 8 #80
batch_size = 128
nupdates = 100

for update in xrange(nupdates):

    trajectories = run_policy(env, policy, episodes=episode_size)

    add_value(trajectories, val_func)
    add_gae(trajectories)
    observes, actions, advantages, returns = build_train_set(trajectories)

    pol_loss, pol_kl, pol_entropy = policy.update(observes, actions, advantages, batch_size=batch_size)  
    vf_loss = val_func.fit(observes, returns,batch_size=batch_size)
    
    mean_ret = np.mean([np.sum(t['rewards']) for t in trajectories])
    if (update%1) == 0 or update == (nupdates-1) :
        print '[{}/{}] Mean Ret : {:.3f}, Value Loss : {:.3f}, Policy loss : {:.5f}, Policy KL : {:.5f}, Policy Entropy : {:.3f} ***'.format(
                            (update+1), nupdates, mean_ret, vf_loss, pol_loss, pol_kl, pol_entropy)


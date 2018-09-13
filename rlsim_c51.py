from __future__ import absolute_import
import distdeepq
import datetime
# import gym
import numpy as np
import random
import numpy as np
import scipy.signal
import rlsim_env
import tensorflow as tf
from collections import deque
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

def exp(env_name='straight_4lane_obs_cam',
        lr=1e-4,
        eps=0.0003125,
        max_timesteps=25e6,
        buffer_size=1e6,
        batch_size=32,
        exp_t1=1e6,
        exp_p1=0.1,
        exp_t2=25e6,
        exp_p2=0.01,
        train_freq=4,
        learning_starts=5e4,
        target_network_update_freq=1e4,
        gamma=0.99,
        num_cpu=50,
        dist_params={'Vmin': -10, 'Vmax': 10, 'nb_atoms': 51},
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        #action_res=None
        ):

    env = rlsim_env.make(env_name)
    # env = wrapper_car_racing(env)  # frame stack
    logger.configure(dir=os.path.join('.', datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f")))
    # logger.configure()

    n_action, action_map = discrete_action(env, env_name) # , action_res=action_res)

    model = distdeepq.models.cnn_to_dist_mlp(
        convs=convs, # [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=hiddens, # [512],
        # n_action=n_action,
        dueling=False
    )
    act = distdeepq.learn(
        env,
        p_dist_func=model,
        lr=lr,  # 1e-4
        eps=eps,
        max_timesteps=int(max_timesteps), # 25M
        buffer_size=int(buffer_size), # 1M
        batch_size=int(batch_size),
        exp_t1=exp_t1,
        exp_p1=exp_p1,
        exp_t2=exp_t2,
        exp_p2=exp_p2,
        train_freq=train_freq,
        learning_starts=learning_starts, # 50000
        target_network_update_freq=target_network_update_freq, # 10000
        gamma=gamma,
        num_cpu=num_cpu,
        prioritized_replay=False,
        dist_params=dist_params,
        n_action=int(n_action),
        action_map=action_map
    )
    act.save("car_racing_model.pkl")
   # env.close()


#def get_action_information(env, env_name, action_res=None):
#    action_map = []
#    if isinstance(env.action_space, gym.spaces.Box):
#        if env_name == "CarRacing-v0":
#            action_map = np.zeros([np.prod(action_res), 3])
#            ste = np.linspace(env.action_space.low[0], env.action_space.high[0], num=action_res[0])  # -1~1
#            gas = np.linspace(env.action_space.low[1], env.action_space.high[1], num=action_res[1])  # 0~1
#           brk = np.linspace(env.action_space.low[2], env.action_space.high[2], num=action_res[2])  # 0~1
#            for i in range(action_res[0]):
#                for j in range(action_res[1]):
#                    for k in range(action_res[2]):
#                        s = action_res[2] * action_res[1] * i + action_res[2] * j + k
#                        action_map[s, :] = [ste[i], gas[j], brk[k]]
#        n_action = np.prod(action_res)
#
#    else:
#        raise NotImplementedError("action space not supported")
#
#    return n_action, action_map

def discrete_action(env, env_name):
    
    n_action = 6 # env.action_space
    action_map = np.zeros([n_action,3])
 
    for action in xrange(n_action):
        steering = action%3
        accel = action//3

        if steering == 0:
            angle_cmd = -1.0
        elif steering == 1:
            angle_cmd = 0.0
        else:
            angle_cmd = 1.0
        
        if accel==0:
            pedal_cmd = 0.4
            break_cmd = 0
        elif accel==1:
            pedal_cmd = 0.6 
            break_cmd = 0 
        else:
            pedal_cmd = 0.5
            break_cmd = 0 

        action_map[action, :] = [angle_cmd,pedal_cmd,break_cmd]
 
    return n_action, action_map

if __name__ == '__main__':
    # exp(lr=2.5e-4, max_timesteps=2.5e6, buffer_size=1e4, exp_t1=1e6, exp_t2=2.5e6,
        # exp_p1=0.1, exp_p2=0.01, hiddens=[256],
        # learning_starts=1e4, target_network_update_freq=1e3, num_cpu=4)
    exp(lr=2.5e-4, max_timesteps=5e4, buffer_size=1e4, exp_t1=1e4, exp_t2=2.5e4,  # 1, 5  
        exp_p1=0.1, exp_p2=0.01, hiddens=[256],
        learning_starts=5e2, target_network_update_freq=2e2, num_cpu=16)
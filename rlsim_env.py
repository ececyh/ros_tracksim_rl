import rospy
import math
import numpy as np
import cv2
import time
import random
import threading
from cv_bridge import CvBridge, CvBridgeError

from dbw_mkz_msgs.msg import SteeringCmd, BrakeCmd, GearCmd, ThrottleCmd, TurnSignalCmd
from dbw_mkz_msgs.msg import Gear

from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import SetModelState, SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import Image, Imu, CompressedImage
from std_msgs.msg import Int32, Float32

def make(env_name):
    if env_name == "straight_2lane":
        return straight_2lane_env()
    elif env_name == "straight_2lane_disc":
        return straight_2lane_disc_env()
    elif env_name == "straight_2lane_obs":
        return straight_2lane_obs_env()

    elif env_name == "straight_4lane":
        return straight_4lane_env()
    elif env_name == "straight_4lane_obs":
        return straight_4lane_obs_env()
    elif env_name == "straight_4lane_cam":
        return straight_4lane_cam_env()
    elif env_name == "straight_4lane_obs_cam":
        return straight_4lane_obs_cam_env()


    elif env_name == "straight_4lane_long":
        return straight_4lane_long_env()
    elif env_name == "straight_4lane_long_cam":
        return straight_4lane_long_cam_env()

    else:
        return None


class straight_4lane_env(object):
    def __init__(self):
        rospy.init_node("rl_sim_env")
        #self.initial_model_state = None
        rospy.wait_for_service("/gazebo/set_model_state")

        self.set_state_proxy = rospy.ServiceProxy("/gazebo/set_model_state",SetModelState)

        self.car_name = "vehicle"

        self.init_pose = Pose(Point(-180,-2.3,0.1),Quaternion(0,0,0,1))

        self.init_state = ModelState()
        self.init_state.model_name = self.car_name
        self.init_state.pose = self.init_pose

        self.steering = SteeringCmd()
        self.steering.enable = True
        self.steering.steering_wheel_angle_cmd = 0

        self.accel = ThrottleCmd()
        self.accel.enable = True
        self.accel.pedal_cmd = 0
        self.accel.pedal_cmd_type = 2

        self.brake = BrakeCmd()
        self.brake.enable = True
        self.brake.pedal_cmd = 0
        self.brake.pedal_cmd_type = 2

        self.gear = GearCmd()
        self.gear.cmd.gear = 1

        self.pub_steering = rospy.Publisher('vehicle/steering_cmd', SteeringCmd, queue_size=10) # red car : vehicle
        self.pub_accel = rospy.Publisher('vehicle/throttle_cmd', ThrottleCmd, queue_size=10)
        self.pub_brake = rospy.Publisher('vehicle/brake_cmd', BrakeCmd, queue_size=10)
        self.pub_gear = rospy.Publisher('vehicle/gear_cmd', GearCmd, queue_size=10)

        self.bridge = CvBridge()

        self.feature_state = np.zeros(10)

        self.deviation = 0
        self.reach_goal = False
        self.out_of_lane = False
        self.x_coord = self.init_pose.position.x
        self.goal_x = 180

        self.state_sub = rospy.Subscriber('gazebo/model_states',ModelStates,self.state_callback)
        self.lane_sub = rospy.Subscriber('/vehicle/lane_number',Int32, self.lane_callback)
        self.deviation_sub = rospy.Subscriber('/vehicle/deviation',Float32, self.deviation_callback)

        self.observation_space = 8
        self.action_space = 3

        self.time_limit = 100
        self.time = 0

    def lane_callback(self, data):
        self.lane_number = data.data
        self.out_of_lane = (self.lane_number == -1)

    def deviation_callback(self,data):
        self.deviation = abs(data.data)

    def state_callback(self, data):

        '''
        if self.initial_model_state is None:
            self.initial_model_state = data
        '''
        idx = data.name.index(self.car_name)
        pose = data.pose[idx].position
        ori = data.pose[idx].orientation
        lin = data.twist[idx].linear
        
        if (pose.x > self.goal_x and not self.out_of_lane) or self.reach_goal:
            self.reach_goal = True
        else:
            self.reach_goal = False

        self.feature_state = np.asarray([pose.x,pose.y,ori.x,ori.y,ori.z,ori.w,lin.x,lin.y])

    def render(self):
        print(self.feature_state)
        return

    def reset(self):
        '''
        if self.initial_model_state is not None:
            for e,names in enumerate(self.initial_model_state.name):
                obj_state = ModelState()
                obj_state.model_name = names
                obj_state.pose = self.initial_model_state.pose[e]
                obj_state.twist = self.initial_model_state.twist[e]
                self.set_state_proxy(obj_state)
        '''
        while(True):
            check = self.set_state_proxy(self.init_state)

            rospy.sleep(0.01)
            time.sleep(0.01)
            if check and not self.out_of_lane:
                x_error = abs(self.init_pose.position.x - self.feature_state[0])
                y_error = abs(self.init_pose.position.y - self.feature_state[1])
                if x_error + y_error < 0.5:
                    break

        self.reach_goal = False
        self.out_of_lane = False

        self.x_coord = self.init_pose.position.x

        next_obsv = self.feature_state

        return next_obsv

    def step(self, action):
        
        self.time += 1

        self.steering.steering_wheel_angle_cmd = action[0]
        self.accel.pedal_cmd = action[1]
        self.brake.pedal_cmd = action[2]
        self.gear.cmd.gear = 4

        rate = rospy.Rate(10)
        for i in range(5):
            self.pub_steering.publish(self.steering)
            self.pub_accel.publish(self.accel)
            self.pub_brake.publish(self.brake)
            self.pub_gear.publish(self.gear)

            rate.sleep()

        next_obsv = self.feature_state

        reward = self.reward()
        done, success = self.is_terminated()

        if done:
            self.time = 0

        return next_obsv, reward, done, success

    def is_terminated(self):

        if self.time == self.time_limit:
            return True, False

        if self.out_of_lane:
            
            return True, False

        if self.reach_goal:
            return True, True
        else:
            return False, False

    def reward(self):

        movement = self.feature_state[0] - self.x_coord
        self.x_coord = self.feature_state[0]

        dev = self.deviation
        
        weighted_reward = 0.1 * movement + 10 * self.reach_goal - 5 * self.out_of_lane - 1 * dev
        #print ("m :{:.2f}, r :{}, o :{}, d :{:.2f}, wr: {:.2f}".format(movement, self.reach_goal, self.out_of_lane, dev, weighted_reward))
        
        return weighted_reward

class straight_4lane_cam_env(straight_4lane_env):
    def __init__(self):
        super(straight_4lane_cam_env, self).__init__()

        self.img_state = np.zeros([800,800,3])
        self.img_state_stacked = []

        self.cam_sub = rospy.Subscriber('vehicle/front_camera/image_raw',Image,self.cam_state_callback)

        self.observation_space = (800,800,3)
        self.action_space = 3

        self.time_limit = 100
        self.time = 0

    def cam_state_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        (raws,cols,channels) = cv_image.shape

        self.img_state = cv_image

    def render(self):
        cv2.imshow("camera image", self.img_state)
        cv2.waitKey(1)

    def reset(self):
        super(straight_4lane_cam_env, self).reset()

        cur_img = self.img_state
        self.img_state_stacked = [cur_img]*4
        next_obsv = np.concatenate(self.img_state_stacked, axis=2)

        return next_obsv

    def step(self, action):
        
        next_feature, reward, done, success = super(straight_4lane_cam_env, self).step(action)

        cur_img = self.img_state

        self.img_state_stacked.pop(0)
        self.img_state_stacked.append(cur_img)

        next_obsv = np.concatenate(self.img_state_stacked, axis=2)

        return next_obsv, reward, done, success

class straight_4lane_obs_env(straight_4lane_env):
    def __init__(self):
        self.obs_list = []

        rd = [range(4),range(4)]
        random.shuffle(rd[0])
        random.shuffle(rd[1])
        rd = rd[0] + rd[1] 

        for i in range(8):
            model_state = ModelState()
            model_state.model_name = 'mkz'+str(i)

            x = -150 + 40*i + random.uniform(-10,10)
            y = -7.5 + 5*rd[i]

            model_state.pose = Pose(Point(x,y,0.1),Quaternion(0,0,0,1))

            self.obs_list.append({'init_state':model_state, 'name':model_state.model_name, 'init_x':x, 'init_y':y, \
'pub_steering':rospy.Publisher('mkz'+str(i)+'/steering_cmd', SteeringCmd, queue_size=10), \
'pub_accel':rospy.Publisher('mkz'+str(i)+'/throttle_cmd', ThrottleCmd, queue_size=10), \
'pub_brake':rospy.Publisher('mkz'+str(i)+'/brake_cmd', BrakeCmd, queue_size=10), \
'pub_gear':rospy.Publisher('mkz'+str(i)+'/gear_cmd', GearCmd, queue_size=10)})

        self.repulse = 0

        super(straight_4lane_obs_env, self).__init__()

        self.collision = False
        self.collision_sub = rospy.Subscriber('/vehicle/bumper_state',ContactsState,self.bumpercallback)

        self.init_pose.position.y = -7.5 + 5 * random.randrange(4)

        self.observation_space = 8 + 8*2
        self.action_space = 3

        self.reset_trigger = False

    def bumpercallback(self,data):
        if data.states:
            self.collision = True

    def state_callback(self,data):

        '''
        if self.initial_model_state is None:
            self.initial_model_state = data
        '''

        idx = data.name.index(self.car_name)
        pose = data.pose[idx].position
        ori = data.pose[idx].orientation
        lin = data.twist[idx].linear

        if (pose.x > self.goal_x and not self.out_of_lane) or self.reach_goal:
            self.reach_goal = True
        else:
            self.reach_goal = False

        obs_features = []

        cur_rep = 0
        for n, obs in enumerate(self.obs_list):
            obs_idx = data.name.index(obs['name'])
            obs_pose = data.pose[obs_idx].position

            obs_x = obs_pose.x - pose.x
            obs_y = obs_pose.y - pose.y
            obs_features += [obs_x, obs_y]

            if abs(obs_y)<2.5:
                if abs(obs_x)<5:
                    cur_rep += 1 + 0.8*(5-abs(obs_x))
                elif abs(obs_x)<10:
                    cur_rep += 0.2*(10-abs(obs_x))
                elif abs(obs_x)<15 and lin.x > 15:
                    cur_rep += 0.5
                elif abs(obs_x)<20 and lin.x > 20:
                    cur_rep += 0.2
                elif abs(obs_x)<25 and lin.x > 25:
                    cur_rep += 0.1
                elif lin.x > 30:
                    cur_rep += 0.1

        if cur_rep != 0:
            self.repulse += 0.005 * cur_rep
        else:
            self.repulse = 0

        self.feature_state = np.asarray([pose.x,pose.y,ori.x,ori.y,ori.z,ori.w,lin.x,lin.y] + obs_features)

    def obs_control_thread(self):

        if self.reset_trigger:
            return

        for n, obs in enumerate(self.obs_list):
            obs_x = self.feature_state[8 + 2*n] + self.feature_state[0]
            obs_y = self.feature_state[8 + 2*n+1] + self.feature_state[1]
            if obs_x > self.goal_x: # and not obs['trigger']:
                #obs['trigger'] = True
                model_state = ModelState()
                model_state.model_name = obs['name']

                x = -190
                y = -7.5 + 5 * random.randrange(4)

                model_state.pose = Pose(Point(x,y,0.1),Quaternion(0,0,0,1))
                #self.obs_list[n]['init_x'] = x
                #self.obs_list[n]['init_y'] = y

                try:
                    self.set_state_proxy(model_state)
                except:
                    continue
                
            else:
                model_state = ModelState()
                model_state.model_name = obs['name']

                x = obs_x + random.uniform(0.2,0.3)
                y = obs_y

                model_state.pose = Pose(Point(x,y,0.1),Quaternion(0,0,0,1))
                model_state.twist.linear.x = random.uniform(5.0,6.0)

                #self.obs_list[n]['init_x'] = x
                try:
                    self.set_state_proxy(model_state)
                except:
                    continue
                
        threading.Timer(0.1,self.obs_control_thread).start()

    def reset(self):

        '''
        if self.initial_model_state is not None:
            for e,names in enumerate(self.initial_model_state.name):
                obj_state = ModelState()
                obj_state.model_name = names
                obj_state.pose = self.initial_model_state.pose[e]
                obj_state.twist = self.initial_model_state.twist[e]
                self.set_state_proxy(obj_state)
        '''

        self.reset_trigger = True

        rd = [range(4),range(4)]
        random.shuffle(rd[0])
        random.shuffle(rd[1])
        rd = rd[0] + rd[1] 

        for n, obs in enumerate(self.obs_list):
            model_state = ModelState()
            model_state.model_name = 'mkz'+str(n)

            x = -150 + 40*n + random.uniform(-10,10)
            y = -7.5 + 5 * rd[n]

            model_state.pose = Pose(Point(x,y,0.1),Quaternion(0,0,0,1))
            self.obs_list[n]['init_state'] = model_state
            self.obs_list[n]['init_x'] = x
            self.obs_list[n]['init_y'] = y

            while(True):
                check = self.set_state_proxy(model_state)

                rospy.sleep(0.01)
                if check:
                    obs_x = self.feature_state[8 + 2*n] + self.feature_state[0]
                    obs_y = self.feature_state[8 + 2*n+1] + self.feature_state[1]

                    x_error = abs(x - obs_x)
                    y_error = abs(y - obs_y)
                    if x_error + y_error < 0.5:
                        break

        self.init_pose.position.y = -7.5 + 5 * random.randrange(4)

        while(True):
            check = self.set_state_proxy(self.init_state)

            rospy.sleep(0.01)
            if check and not self.out_of_lane:
                x_error = abs(self.init_pose.position.x - self.feature_state[0])
                y_error = abs(self.init_pose.position.y - self.feature_state[1])
                if x_error + y_error < 0.5:
                    break

        self.reach_goal = False
        self.out_of_lane = False

        self.x_coord = self.init_pose.position.x
        self.x_coord = self.init_pose.position.x

        self.repulse = 0
        self.collision = False
        next_obsv = self.feature_state

        self.reset_trigger = False
        self.obs_control_thread()

        return next_obsv

    def step(self, action):
        
        self.time += 1

        self.steering.steering_wheel_angle_cmd = action[0]
        self.accel.pedal_cmd = action[1]
        self.brake.pedal_cmd = action[2]
        self.gear.cmd.gear = 4

        rate = rospy.Rate(10)
        for i in range(5):
            self.pub_steering.publish(self.steering)
            self.pub_accel.publish(self.accel)
            self.pub_brake.publish(self.brake)
            self.pub_gear.publish(self.gear)

            rate.sleep()

        next_obsv = self.feature_state

        reward = self.reward()
        done, success = self.is_terminated()

        if done:
            self.time = 0

        return next_obsv, reward, done, success

    def is_terminated(self):

        if self.time == self.time_limit:
            return True, False

        if self.out_of_lane:
            return True, False

        if self.collision:
            return True, False

        if self.reach_goal:
            return True, True
        else:
            return False, False

    def reward(self):

        movement = self.feature_state[0] - self.x_coord
        self.x_coord = self.feature_state[0]

        dev = self.deviation

        rep = max(5,self.repulse)

        weighted_reward = 0.1 * movement + 10 * self.reach_goal - 5 * self.out_of_lane - 1 * dev - 5 * self.collision - 1 * rep
        #print ("m :{:.2f}, r :{}, o :{}, d :{:.2f}, wr: {:.2f}".format(movement, self.reach_goal, self.out_of_lane, dev, weighted_reward))
        
        return weighted_reward


class straight_4lane_obs_cam_env(straight_4lane_obs_env):
    def __init__(self):
        super(straight_4lane_obs_cam_env, self).__init__()

        self.img_state = np.zeros([800,800,3])
        self.img_state_stacked = []

        self.cam_sub = rospy.Subscriber('vehicle/front_camera/image_raw',Image,self.cam_state_callback)

        self.observation_space = (800,800,3)
        self.action_space = 3

        self.time_limit = 100
        self.time = 0

    def cam_state_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)

        (raws,cols,channels) = cv_image.shape

        self.img_state = cv_image

    def render(self):
        cv2.imshow("camera image", self.img_state)
        cv2.waitKey(1)

    def reset(self):
        super(straight_4lane_obs_cam_env, self).reset()

        cur_img = self.img_state
        self.img_state_stacked = [cur_img]*4
        next_obsv = np.concatenate(self.img_state_stacked, axis=2)

        return next_obsv

    def step(self, action):
        
        next_feature, reward, done, success = super(straight_4lane_obs_cam_env, self).step(action)

        cur_img = self.img_state

        self.img_state_stacked.pop(0)
        self.img_state_stacked.append(cur_img)

        next_obsv = np.concatenate(self.img_state_stacked, axis=2)

        return next_obsv, reward, done, success


class straight_4lane_long_env(straight_4lane_env):
    def __init__(self):
        super(straight_4lane_long_env, self).__init__()

        self.init_pose1.position.x = -400
        self.init_pose2.position.x = -400

        self.init_state1.pose = self.init_pose1
        self.init_state2.pose = self.init_pose2
        self.x_coord = self.init_pose1.position.x

        self.goal_x = 350
        self.time_limit = 200
        self.time = 0

class straight_4lane_long_cam_env(straight_4lane_cam_env):
    def __init__(self):
        super(straight_4lane_long_cam_env, self).__init__()
        self.init_pose1.position.x = -400
        self.init_pose2.position.x = -400

        self.init_state1.pose = self.init_pose1
        self.init_state2.pose = self.init_pose2
        self.x_coord = self.init_pose1.position.x

        self.goal_x = 350
        self.time_limit = 200
        self.time = 0

if __name__ == "__main__":
    print('start')
    env = make("straight_4lane_obs_cam")

    for t in range(100):
        env.reset()
        for i in range(200):
            env.render()
            #print(i)
            s,r,d,i = env.step([0.0,0.5,0])
            if d:
                break


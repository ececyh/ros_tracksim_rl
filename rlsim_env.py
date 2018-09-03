import rospy
import math
import numpy as np
import cv2
import time
from cv_bridge import CvBridge, CvBridgeError

from dbw_mkz_msgs.msg import SteeringCmd, BrakeCmd, GearCmd, ThrottleCmd, TurnSignalCmd
from dbw_mkz_msgs.msg import Gear

from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SetModelState
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

    else:
        return None


class straight_4lane_env(object):
    def __init__(self):
        rospy.init_node("rl_sim_env")
        self.initial_model_state = None
        rospy.wait_for_service("/gazebo/set_model_state")

        self.set_state_proxy = rospy.ServiceProxy("/gazebo/set_model_state",SetModelState)

        self.car_name1 = "vehicle"
        self.car_name2 = "mkz"

        self.init_pose1 = Pose()
        self.init_pose1.position.x = -180
        self.init_pose1.position.y = -2.3
        self.init_pose1.position.z = 0.1
        self.init_pose1.orientation.x = 0
        self.init_pose1.orientation.y = 0
        self.init_pose1.orientation.z = 0
        self.init_pose1.orientation.w = 1

        self.init_pose2 = Pose()
        self.init_pose2.position.x = -180
        self.init_pose2.position.y = 2.3
        self.init_pose2.position.z = 0.1
        self.init_pose2.orientation.x = 0
        self.init_pose2.orientation.y = 0
        self.init_pose2.orientation.z = 0
        self.init_pose2.orientation.w = 1

        self.init_state1 = ModelState()
        self.init_state1.model_name = self.car_name1
        self.init_state1.pose = self.init_pose1

        self.init_state2 = ModelState()
        self.init_state2.model_name = self.car_name2
        self.init_state2.pose = self.init_pose2

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
        self.parking_gear = rospy.Publisher('mkz/gear_cmd', GearCmd, queue_size=10)

        self.bridge = CvBridge()

        self.feature_state = np.zeros(10)

        self.deviation = 0
        self.reach_goal = False
        self.out_of_lane = False
        self.x_coord = self.init_pose1.position.x

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

        if self.initial_model_state is None:
            self.initial_model_state = data

        idx = data.name.index(self.car_name1)
        pose = data.pose[idx].position
        ori = data.pose[idx].orientation
        lin = data.twist[idx].linear
        
        if (pose.x >180 and not self.out_of_lane) or self.reach_goal:
            self.reach_goal = True
        else:
            self.reach_goal = False

        self.feature_state = np.asarray([pose.x,pose.y,ori.x,ori.y,ori.z,ori.w,lin.x,lin.y])

    def render(self):
        return

    def reset(self):
        if self.initial_model_state is not None:
            for e,names in enumerate(self.initial_model_state.name):
                obj_state = ModelState()
                obj_state.model_name = names
                obj_state.pose = self.initial_model_state.pose[e]
                obj_state.twist = self.initial_model_state.twist[e]
                self.set_state_proxy(obj_state)

        while(True):
            check_1 = self.set_state_proxy(self.init_state1)
            check_2 = self.set_state_proxy(self.init_state2)

            rospy.sleep(0.1)
            time.sleep(0.1)
            if check_1 and check_2:
                break

        self.gear.cmd.gear = 1
        self.parking_gear.publish(self.gear)
        self.gear.cmd.gear = 4

        self.reach_goal = False
        self.out_of_lane = False

        self.x_coord = self.init_pose1.position.x

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


if __name__ == "__main__":

    env = make("straight_4lane_cam")
    env.reset()

    env.step([0,1,0])

    for t in range(30):
        env.reset()
        for i in range(50):
            env.render()
            s,r,d,i = env.step([0,1.0,0])
            if d:
                break


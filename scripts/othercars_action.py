#!/usr/bin/env python
# -*- coding: utf-8 -*-

import roslib
roslib.load_manifest('dbw_runner')

import rospy
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
import math



if __name__== '__main__':
    rospy.init_node('othercars_client')

    # path ( x, y, yaw)
    path = [(-175.0, 2.3, 0), (-170.0, 2.3, 0), (-165.0, 2.3, 0), (-160.0, 0, -0.875), (-155.0, -2.3, 0),
            (-150.0, -2.3, 0), (-145.0, -2.3, 0), (-140, -2.3, 0), (-135, -2.3, 0), (-130, -2.3, 0)]

    rospy.wait_for_service('/gazebo/set_model_state')
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    rospy.wait_for_service('/gazebo/get_model_state')
    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    while not rospy.is_shutdown():
        try :
            get = get_state('mkz','world')
        except rospy.ServiceException, e:
            continue

        break


    state = ModelState()

    state.model_name = "mkz"

    iterate_rate = rospy.Rate(0.5) # 2초에 한번씩 반복
    while not rospy.is_shutdown():
        step = 0.1 # 0.1 만큼 x 움직임
        step_rate = rospy.Rate(50)
        for i in range(len(path) - 1):
            # 두 점 사이의 line
            x_delta = path[i+1][0] - path[i][0]
            y_delta = path[i+1][1] - path[i][1]
            w_delta = path[i+1][2] - path[i][2]

            slope_pos = y_delta / x_delta
            slope_yaw = w_delta / x_delta

            for j in range(int(abs(x_delta/step))):
                pose = Pose()
                pose.position.x = path[i][0] + j*step
                pose.position.y = slope_pos*(j*step) + path[i][1]
                pose.position.z = 0.3


                yaw = slope_yaw*(j*step) + path[i][2]
                cy = math.cos(yaw * 0.5)
                sy = math.sin(yaw * 0.5)
                cr = cp = 1
                sr = sp = 0

                pose.orientation.z = sy * cr * cp - cy * sr * sp
                pose.orientation.w = cy * cr * cp + sy * sr * sp

                state.pose = pose
                rospy.loginfo("Moving Car")
                ret = set_state(state)
                step_rate.sleep()


        iterate_rate.sleep()




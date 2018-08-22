#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty as EmptySrv
from std_msgs.msg import Empty as EmptyMsg


import math
import rosnode
from geometry_msgs.msg import Twist


class car_position(object):

    car_pose = ()
    data = PoseStamped
    def __init__(self):
        position_subscriber = rospy.Subscriber("gazebo/model_states", ModelStates, self.positioncallback)


    def positioncallback(self,data):  # position data가 들어올 때마다 실행
        # 각종 variable 계산

        if 'mkz' in data.name :
            i = data.name.index('mkz')

            qw = data.pose[i].orientation.w
            qz = data.pose[i].orientation.z
            qx = data.pose[i].orientation.x
            qy = data.pose[i].orientation.y

            #print data.pose[i].position
            yaw = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
            #print "yaw", yaw

            self.car_pose = (data.pose[i].position.x, data.pose[i].position.y, yaw)


def distance(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)


if __name__ == '__main__':

    rospy.init_node('othercars', anonymous=True)  # node 시작

    pathpublisher= rospy.Publisher('/mkz/target_path',Path,queue_size=10)
    pub = rospy.Publisher('/mkz/cmd_vel', Twist, queue_size=10)

    path = [(-175.0, 2.3, 0), (-170.0, 2.3, 0), (-165.0, 2.3, 0), (-160.0, 0, -0.875), (-155.0, -2.3, 0),
            (-150.0, -2.3, 0), (-145.0, -2.3, 0), ( -140, -2.3, 0), ( -135, -2.3, 0), ( -130, -2.3, 0)]

    car_pose = car_position()

    count = 0
    print "waiting for car position"
    while not car_pose.car_pose :
        count += 1

    print "done! start running!" + '\n'

    #print "start control"
    rate = rospy.Rate(1)
    while not rospy.is_shutdown() :
        msg = Path()
        msg.header.frame_id = 'mkz/base_footprint'


        # print "직진"
        # for i in range(10):
        #     pose = PoseStamped()
        #     pose.pose.position.x = i*5
        #     pose.header.frame_id = 'base_footprint'
        #     msg.poses.append(pose)



        #while문으로 가장 가까운 점 찾기  path relative 업데이트

        min_d = float('inf')
        idx = 0
        deg = 0
        #print 'car_pose : ', car_pose.car_pose[0], car_pose.car_pose[1], car_pose.car_pose[2]

        for path_point in path :
            d = distance(car_pose.car_pose, path_point)
            deg = car_pose.car_pose[2] - math.atan2(path_point[1],path_point[0])

            if min_d > d and {deg < math.pi/8 or deg > -math.pi/8}:
                min_d = d
                idx = path.index(path_point)


        if idx == (len(path)-1-1) and min_d < 2.0 :


            print "reached goal. termintate"
            rosnode.kill_nodes(['/mkz/path_following'])


            stop_msg = Twist()
            stop_msg.linear.x = 0
            stop_msg.angular.z = 0

            r = rospy.Rate(50)
            for i in range(100):
                pub.publish(stop_msg)
                r.sleep()
            break

        size_path_relative = len(path) - idx

        #print 'closest point : ', idx, " ", path[idx]
        # print 'min_d : ', min_d
        # print '\n'

        msg_path = []
        for i in range(size_path_relative):
            p = path[idx+i]
            theta = math.atan2(p[1]-car_pose.car_pose[1],p[0]-car_pose.car_pose[0])
            d = distance(p,car_pose.car_pose)

            point = (d*math.cos(theta - car_pose.car_pose[2]), d*math.sin(theta - car_pose.car_pose[2]),p[2]-car_pose.car_pose[2])

            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]

            yaw = point[2]
            cy = math.cos(yaw*0.5)
            sy = math.sin(yaw*0.5)
            cr = cp = 1
            sr = sp = 0

            pose.pose.orientation.z = sy*cr*cp - cy*sr*sp
            pose.pose.orientation.w = cy*cr*cp + sy*sr*sp

            msg.poses.append(pose)

            msg_path.append(point)

        #print 'path :', msg_path
        #rate.sleep()
        pathpublisher.publish(msg)


    # reset
    pause_physics = rospy.ServiceProxy('/gazebo/pause_physics',EmptySrv)
    unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', EmptySrv)
    set_state = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)

    rospy.loginfo("Pausing physics")
    pause_physics()

    initial_pose = Pose()
    initial_pose.position.x = -180.0
    initial_pose.position.y = 2.3
    initial_pose.position.z = 0.3
    initial_pose.orientation.x = 0
    initial_pose.orientation.y = 0
    initial_pose.orientation.z = 0
    initial_pose.orientation.w = 0

    state = ModelState()

    state.model_name = "mkz"
    state.pose = initial_pose


    rospy.loginfo("Moving physics")
    ret = set_state(state)

    rospy.loginfo("unPausing physics")
    unpause_physics()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, NavSatFix  # gps msg type
import numpy as np
from gazebo_msgs.msg import ModelStates
from matplotlib import path
import Carinfo
from math import atan2
from geometry_msgs.msg import Twist


MAX_DIST = 40


class Trackinfo(object):

    #--------------------------------
    my_car = Carinfo.Carinfo()

    deviation = -1       # [left, right]
    distance = 0               # 달려온거리
    heading = 0                # line에 대한 heading, 기울기
    steering = 0
    lane_number = 0            # 자동차가 있는 lane number, start from 0
    current_segment = 0

    Track = []

    class obstacles(object):    # 장애물이 되는 point들
        def __init__(self):
            self.left = [[], []]
            self.right = [[], []]
            self.middle = [[], []]

    class dist2obstacle(object):        # 장애물까지의 거리 [front, back]
        left = [MAX_DIST, MAX_DIST]     # left lane의 앞뒤 장애물까지의 거리
        middle = [MAX_DIST, MAX_DIST]   # 자동차가 있는 lane
        right = [MAX_DIST, MAX_DIST]    # right lane

    dist2obstacles = dist2obstacle()
    #-----------------------------------

    def __init__(self):
        position_subscriber = rospy.Subscriber("gazebo/model_states", ModelStates, self.positioncallback)
        sensor_subscriber = rospy.Subscriber("vehicle/velodyne_points", PointCloud2, self.sensorcallback)
        command_subscriber = rospy.Subscriber('/vehicle/cmd_vel', Twist, self.cmdcallback)

        self.my_car.width = rospy.get_param("/car/width", 3.0)
        self.my_car.height = rospy.get_param("/car/height", 3.0)
        self.my_car.mass = rospy.get_param("/car/mass", 3.0)


    # subscriber callback
    def sensorcallback(self, data): # sensor data가 들어올 때마다 실행
        if self.my_car.pose: self.findObstacles(data) # obstacle을 찾아 dist2obstacle 저장


    def positioncallback(self, data):  # position data가 들어올 때마다 실행
        # 각종 variable 계산
        if not ('vehicle' in data.name):  # vehicle 이 gazebo 상에서 아직 생성되지 않은 경우
            return
        else:
            i = data.name.index('vehicle')

            qw = data.pose[i].orientation.w
            qz = data.pose[i].orientation.z
            qx = data.pose[i].orientation.x
            qy = data.pose[i].orientation.y

        # print data.pose[i].position
        yaw = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

        self.my_car.pose = (data.pose[i].position.x + 1.1*np.cos(yaw),
                            data.pose[i].position.y + 1.1*np.sin(yaw), yaw )
        self.my_car.velocity = data.twist[i]
        self.current_segment = self.findSegment(self.my_car.pose[0:2])
        self.lane_number = self.findLane(self.my_car.pose[0:2])
        self.findSidelines(self.my_car.pose[0:2])
        self.findDeviation(self.my_car.pose)
        self.findHeading(self.my_car.pose)

    def cmdcallback(self,data):

        self.steering = data.angular.z


    # setting
    def set_world(self, seglist):
        self.Track = seglist

    # function
    def findObstacles(self, data):
        #TODO mindist 계산 방법 바꾸기
        #TODO laser를 조금 바꿔볼까 생각중

        obstaclepoints = Trackinfo.obstacles()

        # point cloud형식으로 들어오는 laser data를 point by point 탐색
        for p in point_cloud2.read_points(data, skip_nans=True):


            # p : laser sensor의 위치가 기준
            #    이 때 laser sensor 의 xy를 자동차의 위치로 설정
            # pose : 지도의 원점이 기준. 절대적 위치.

            #p = (point[0] + 1.1, point[1], point[2] + 1.49)  # 수치 : dbw_mkz_gazebo/urdf/mkz.urdf.xacro
            if p[2] + 1.49 < 0.02 or (- 2.2 < p[0] < 2.8 and abs(p[1]) < 1): continue
            # 장애물뿐만 아니라 차체도 센서에 잡히기 때문에 너무 가까운 p들은 제외 # TODO empirical value
            if np.linalg.norm(p[0:2]) > MAX_DIST : continue

            theta = self.my_car.pose[2]
            pose = (p[0] * np.cos(theta) + p[1] * np.cos(theta + np.pi / 2) + self.my_car.pose[0],
                    p[0] * np.sin(theta) + p[1] * np.sin(theta + np.pi / 2) + self.my_car.pose[1])

            lane_num = self.findLane(pose) # pose를 기반으로 이 점이 어느 lane에 있는지 확인
            # 장애물이 되는 point들을 분류해서 정리
            if lane_num == self.lane_number - 1:  # left
                if p[0] >= 0:
                    obstaclepoints.left[0].append(p)
                else:
                    obstaclepoints.left[1].append(p)

            elif lane_num == self.lane_number:  # middle
                if p[0] >= 0:
                    obstaclepoints.middle[0].append(p)
                else:
                    obstaclepoints.middle[1].append(p)

            elif lane_num == self.lane_number + 1:  # right
                if p[0] >= 0:
                    obstaclepoints.right[0].append(p)
                else:
                    obstaclepoints.right[1].append(p)


        # 모든 obstacle points를 정리하고 난 후 이 중 제일 가까운 점을 골라 dist2obstacle에 저장.
        # # find closest point
        self.dist2obstacles.left[0] = self.findMinDist(obstaclepoints.left[0])
        self.dist2obstacles.left[1] = self.findMinDist(obstaclepoints.left[1])
        self.dist2obstacles.right[0] = self.findMinDist(obstaclepoints.right[0])
        self.dist2obstacles.right[1] = self.findMinDist(obstaclepoints.right[1])
        self.dist2obstacles.middle[0] = self.findMinDist(obstaclepoints.middle[0])
        self.dist2obstacles.middle[1] = self.findMinDist(obstaclepoints.middle[1])
        #
        del obstaclepoints

    def findMinDist(self, l):
        # l : list of points
        minD = MAX_DIST

        if l:
            for point in l:
                d = pow(point[0] ** 2 + point[1] ** 2, 0.5)
                if d < minD: minD = d
        return minD

    def findSegment(self, position):
        # 주어진 position이 몇번째 segment에 있는가.
        for seg in self.Track:
            p = path.Path(seg.boundary)
            if p.contains_points([position]):
                return self.Track.index(seg)

        return -1

    def findLane(self, pose):
        # 주어진 position이 몇번째 lane에 있는가

        seg_idx = self.findSegment(pose)
        if seg_idx == -1:
            return -1
        else:
            seg = self.Track[seg_idx] # pose가 속한 segment


        if seg.type == 'straight':

            lane1vec = np.subtract(seg.lane_pos[0][0:2], seg.lane_pos[1][0:2])
            posevec = np.subtract(pose, seg.lane_pos[1][0:2])

            c = np.cross(lane1vec, posevec)

            devfromline0 = np.linalg.norm(c) / np.linalg.norm(lane1vec)
            lane_number = int(np.floor_divide(devfromline0, seg.lane_width))

        elif seg.type == 'curved':

            vecfromOrigin = np.subtract(pose, seg.origin[0:2])
            devfromOrigin = np.linalg.norm(vecfromOrigin)
            devfromline0 = devfromOrigin - (seg.radius - (seg.nr_lane / 2) * seg.lane_width)
            lane_number = int(np.floor_divide(devfromline0, seg.lane_width))


        if seg.reverted : return seg.nr_lane - lane_number -1
        return lane_number


    def findSidelines(self, pose):
        # pose를 기준으로 양쪽 line을 찾는다.

        seg_idx = self.findSegment(pose)
        if seg_idx == -1:
            self.leftline = [-1, -1]
            self.rightline = [-1, -1]
            return

        else:
            seg = self.Track[seg_idx] #pose가 속한 segment



        if self.lane_number >= self.Track[seg_idx].nr_lane :
            print 'out of the track!'
            return
                                 # line의 시작점                       line의 끝점
        self.leftline = [seg.lane_pos[2 * self.lane_number], seg.lane_pos[2 * self.lane_number + 1]]
        self.rightline = [seg.lane_pos[2 * self.lane_number + 2], seg.lane_pos[2 * self.lane_number + 3]]

    def findDeviation(self, pose):
        # 양쪽 line으로부터 떨어진 거리

        if self.current_segment == -1: return

        if self.Track[self.current_segment].type == 'straight':
            leftline_vec = np.subtract(self.leftline[1], self.leftline[0])
            left_pos_vec = np.subtract(pose[0:2], self.leftline[0][0:2])
            c = np.cross(leftline_vec, left_pos_vec)

            self.deviation = np.linalg.norm(c) / np.linalg.norm(leftline_vec) - self.Track[self.current_segment].lane_width/2# left

        elif self.Track[self.current_segment].type == 'curved':
            curr_seg = self.Track[self.current_segment]
            vecfromOrigin = np.subtract(pose[0:2], curr_seg.origin[0:2])
            devfromOrigin = np.linalg.norm(vecfromOrigin)

            inner_rad = curr_seg.radius - (curr_seg.nr_lane / 2) * curr_seg.lane_width

            if curr_seg.reverted :
                devfromline = devfromOrigin - (inner_rad + (curr_seg.nr_lane - self.lane_number - 1) * curr_seg.lane_width)
                devfromline = curr_seg.lane_width - devfromline

            else :
                devfromline = devfromOrigin - (inner_rad + self.lane_number * curr_seg.lane_width)

            self.deviation = devfromline - curr_seg.lane_width/2



    def findHeading(self, pose):
        # line에 대해 자동차의 기울기

        if self.current_segment == -1: return

        if self.Track[self.current_segment].type == 'straight':

            linevector = np.subtract(self.leftline[1], self.leftline[0])
            linetheta = np.arctan2(linevector[1], linevector[0])

            self.heading = pose[2] - linetheta

            #if self.Track[self.current_segment].reverted == True : self.heading -= np.pi

            if self.heading > np.pi:    self.heading -= 2 * np.pi
            elif self.heading < -np.pi:    self.heading += 2 * np.pi


        elif self.Track[self.current_segment].type == 'curved':

            posevecfromorigin = np.subtract(pose[0:2], self.Track[self.current_segment].origin[0:2])
            tangenttheta = np.arctan2(posevecfromorigin[1], posevecfromorigin[0]) + np.pi / 2 #접선벡터

            if tangenttheta > np.pi:    tangenttheta -= 2 * np.pi
            elif tangenttheta < -np.pi:    tangenttheta += 2 * np.pi

            diff = tangenttheta - pose[2]

            if self.Track[self.current_segment].reverted == True: diff -= np.pi

            if diff > np.pi: diff -= 2 * np.pi
            elif diff < -np.pi: diff += 2 * np.pi


            self.heading = -diff
<<<<<<< HEAD

=======
>>>>>>> e91af4e260aeb0bca272391602c9555d64a32886

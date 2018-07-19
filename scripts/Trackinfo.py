#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, NavSatFix  # gps msg type
import numpy as np
from gazebo_msgs.msg import ModelStates
from matplotlib import path
import Carinfo

MAX_DIST = 40


class Trackinfo(object):

    #--------------------------------
    my_car = Carinfo.Carinfo()

    deviation = [-1, -1]       # [left, right]
    distance = 0               # 달려온거리
    heading = 0                # line에 대한 heading, 기울기
    lane_number = 0            # 자동차가 있는 lane number, start from 0
    leftline = []
    rightline = []
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

        self.my_car.width = rospy.get_param("/car/width", 3.0)
        self.my_car.height = rospy.get_param("/car/height", 3.0)
        self.my_car.mass = rospy.get_param("/car/mass", 3.0)


    # subscriber callback
    def sensorcallback(self, data): # sensor data가 들어올 때마다 실행
        if self.my_car.pose: self.findObstacles(data) # obstacle을 찾아 dist2obstacle 저장


    def positioncallback(self, data):  # position data가 들어올 때마다 실행
        # 각종 variable 계산
        i = data.name.index('vehicle')
        self.my_car.pose = (data.pose[i].position.x, data.pose[i].position.y, data.pose[i].orientation.z)
        self.my_car.velocity = data.twist[i]
        self.current_segment = self.findSegment(self.my_car.pose[0:2])
        self.findSidelines(self.my_car.pose[0:2])
        self.findDeviation(self.my_car.pose)
        self.findHeading(self.my_car.pose)


    # setting
    def set_world(self, seglist):
        self.Track = seglist

    # function
    def findObstacles(self, data):

        obstaclepoints = Trackinfo.obstacles()

        # point cloud형식으로 들어오는 laser data를 point by point 탐색
        for point in point_cloud2.read_points(data, skip_nans=True):

            # point : laser sensor의 위치가 기준. 상대적 위치.
            # p : 자동차의 base_link가 기준. 상대적 위치.
            # pose : 지도의 원점이 기준. 절대적 위치.

            p = (point[0] + 1.1, point[1], point[2] + 1.49)  # 수치 : dbw_mkz_gazebo/urdf/mkz.urdf.xacro
            if p[2] < 0.02 or (- 1.0 < p[0] < 2.4 and abs(p[1]) < 0.9): continue
            # 장애물뿐만 아니라 차체도 laser sensor에 잡히기 때문에 너무 가까운 p들은 제외 # TODO empirical value

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

        lane_number = -1

        if seg.type == 'straight':

            lane1vec = np.subtract(seg.lane_pos[0][0:2], seg.lane_pos[1][0:2])
            posevec = np.subtract(pose, seg.lane_pos[1][0:2])

            c = np.cross(lane1vec, posevec)

            devfromline0 = np.linalg.norm(c) / np.linalg.norm(lane1vec)
            lane_number = int(np.floor_divide(devfromline0, seg.lane_width))

        elif seg.type == 'curved':

            vecfromOrigin = np.subtract(pose, seg.origin[0:2])
            devfromOrigin = np.linalg.norm(vecfromOrigin)
            devfromline0 = devfromOrigin - seg.radius
            lane_number = int(np.floor_divide(devfromline0, seg.lane_width))

        return lane_number


# TODO empirical value
    def findSidelines(self, pose):
        # pose를 기준으로 양쪽 line을 찾는다.

        seg_idx = self.findSegment(pose)
        if seg_idx == -1:
            self.lane_number = -1
            self.leftline = [-1, -1]
            self.rightline = [-1, -1]
            return

        else:
            seg = self.Track[seg_idx] #pose가 속한 segment

        if seg.type == 'straight':

            lane1vec = np.subtract(seg.lane_pos[0][0:2], seg.lane_pos[1][0:2])
            posevec = np.subtract(pose, seg.lane_pos[1][0:2])

            c = np.cross(lane1vec, posevec)

            devfromline0 = np.linalg.norm(c) / np.linalg.norm(lane1vec)
            self.lane_number = int(np.floor_divide(devfromline0, seg.lane_width))

            self.leftline = [seg.lane_pos[2 * self.lane_number], seg.lane_pos[2 * self.lane_number + 1]]
            self.rightline = [seg.lane_pos[2 * self.lane_number + 2], seg.lane_pos[2 * self.lane_number + 3]]

        elif seg.type == 'curved':

            vecfromOrigin = np.subtract(pose, seg.origin[0:2])
            devfromOrigin = np.linalg.norm(vecfromOrigin)
            devfromline0 = devfromOrigin - seg.radius
            self.lane_number = int(np.floor_divide(devfromline0, seg.lane_width))

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

            rightline_vec = np.subtract(self.rightline[1], self.rightline[0])
            right_pos_vec = np.subtract(pose[0:2], self.rightline[0][0:2])
            d = np.cross(rightline_vec, right_pos_vec)

            self.deviation[0] = np.linalg.norm(c) / np.linalg.norm(leftline_vec) # left
            self.deviation[1] = np.linalg.norm(d) / np.linalg.norm(rightline_vec) # right


        elif self.Track[self.current_segment].type == 'curved':
            curr_seg = self.Track[self.current_segment]
            vecfromOrigin = np.subtract(pose[0:2], curr_seg.origin[0:2])
            devfromOrigin = np.linalg.norm(vecfromOrigin)

            devfromline1 = devfromOrigin - (curr_seg.radius + self.lane_number * curr_seg.lane_width)
            devfromline2 = curr_seg.lane_width - devfromline1

            posevecfromorigin = np.subtract(pose[0:2], self.Track[self.current_segment].Origin[0:2])
            tangenttheta = np.arctan2(posevecfromorigin[1], posevecfromorigin[0]) + np.pi / 2  # 접선벡터

            if tangenttheta > np.pi:  tangenttheta -= 2 * np.pi
            elif tangenttheta < -np.pi:   tangenttheta += 2 * np.pi

            diff = tangenttheta - pose[2]
            if diff > np.pi : diff -= 2*np.pi
            elif diff < -np.pi : diff += 2*np.pi

            if abs(diff) <= np.pi/2 :
                self.deviation.left = devfromline1
                self.deviation.right = devfromline2

            else :
                self.deviation.right = devfromline1
                self.deviation.left = devfromline2

            self.deviation[0] = devfromline1 # left
            self.deviation[1] = devfromline2 # right

    def findHeading(self, pose):
        # line에 대해 자동차의 기울기

        if self.current_segment == -1: return

        if self.Track[self.current_segment].type == 'straight':
            linevector = np.subtract(self.leftline[1], self.leftline[0])
            linetheta = np.arctan2(linevector[1], linevector[0])

            self.heading = pose[2] - linetheta

        elif self.Track[self.current_segment].type == 'curved':

            posevecfromorigin = np.subtract(self.pose[0:2], self.Track[self.current_segment].Origin[0:2])
            tangenttheta = np.arctan2(posevecfromorigin[1], posevecfromorigin[0]) + np.pi / 2 #접선벡터

            if tangenttheta > np.pi:    tangenttheta -= 2 * np.pi
            elif tangenttheta < -np.pi:    tangenttheta += 2 * np.pi

            diff = tangenttheta - pose[2]
            if diff > np.pi: diff -= 2 * np.pi
            elif diff < -np.pi: diff += 2 * np.pi

            self.heading = abs(diff)

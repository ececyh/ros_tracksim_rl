# -*- coding: utf-8 -*-
import numpy as np
import csv
import rospy
import os,errno
from geometry_msgs.msg import Twist


class Recorder(object):
    MX_NR_DATA = 5000
    my_car_pose = ()
    nr_data = 0
    filename = ''
    dist = 0
    dist_th = 0.05
    deg_th = 0.05
    directory = ''

    def __init__(self,worldname,hz):

        self.directory = os.path.expanduser("~") + "/data"


        try:
            os.makedirs(self.directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(self.directory):
                pass
            else : raise

        file_number = len(os.listdir(self.directory))
        self.filename = self.directory + "/data" + str(file_number) + '_' + worldname + '_hz' + str(hz) + ".csv"

    def update(self, trackinfo):

        if not trackinfo.my_car.velocity or not trackinfo.my_car.pose: return

        if self.my_car_pose: # 새로 들어온 pose와 이전의 pose 비교
            dist_diff = np.linalg.norm(np.subtract(self.my_car_pose, trackinfo.my_car.pose)[0:2])
            deg_diff = np.linalg.norm(np.subtract(self.my_car_pose, trackinfo.my_car.pose)[2:3])
        else :
            self.my_car_pose = trackinfo.my_car.pose
            dist_diff = 0
            deg_diff = 0

        if (dist_diff > self.dist_th or deg_diff > self.deg_th) and (self.nr_data < self.MX_NR_DATA) and (trackinfo.current_segment is not -1):
            # 일정거리 이상 움직이거나, 일정 각도 이상 움직이고 && 현재 data 수가 maximum data 수를 넘지 않았을 때

            with open(self.filename, 'a') as f:
                velocity = np.linalg.norm([trackinfo.my_car.velocity.linear.x,trackinfo.my_car.velocity.linear.y])
                # data = [trackinfo.dist2obstacles.left[0],trackinfo.dist2obstacles.left[1],
                #         trackinfo.dist2obstacles.middle[0], trackinfo.dist2obstacles.middle[1],
                #         trackinfo.dist2obstacles.right[0], trackinfo.dist2obstacles.right[1],
                #         trackinfo.deviation[0],trackinfo.deviation[1],trackinfo.heading] #data로 저장할 list

                data = [trackinfo.deviation, trackinfo.steering]

                #TODO control msg도 data로 받아 저장할 수 있도록 하기


                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(data)

            self.nr_data += 1
            self.my_car_pose = trackinfo.my_car.pose
            self.dist += dist_diff

    def reset(self):

        file_number = len(os.listdir(self.directory))
        self.filename = self.directory + "/data_" + str(file_number) + ".csv"

        self.my_car_pose = ()
        self.nr_data = 0
        self.dist = 0

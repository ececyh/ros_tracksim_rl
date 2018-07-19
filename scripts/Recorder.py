# -*- coding: utf-8 -*-
import numpy as np
import csv
import rospy
from os import getcwd

from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError
import cv2

bridge = CvBridge()


class Recorder(object):
    MX_NR_DATA = 1000
    my_car_pose = ()
    nr_data = 0
    filename = ''
    dist = 0
    dist_th = 0.1
    deg_th = 0.1
    current_image = 0
    cwd = getcwd()

    def __init__(self, filename):

        self.filename = "data_" + str(int(filename)) + ".csv" #처음에 Recorder class initiate할 때의 시간이 csv 이름
        print "data file name : " + self.filename + "\n"
        image_subscriber = rospy.Subscriber("vehicle/front_camera/image_raw", Image, self.imagecallback)

    def imagecallback(self, data):
        self.current_image = data #camer image가 들어올때마다 저장

    def update(self, trackinfo):

        if self.my_car_pose: # 새로 들어온 pose와 이전의 pose 비교
            dist_diff = np.linalg.norm(np.subtract(self.my_car_pose, trackinfo.my_car.pose)[0:2])
            deg_diff = np.linalg.norm(np.subtract(self.my_car_pose, trackinfo.my_car.pose)[2:3])

        else:
            dist_diff = 100
            deg_diff = 100

        if (dist_diff > self.dist_th or deg_diff > self.deg_th) and (self.nr_data < self.MX_NR_DATA):
            # 일정거리 이상 움직이거나, 일정 각도 이상 움직이고 && 현재 data 수가 maximum data 수를 넘지 않았을 때

            img_filename = 0
            # if self.current_image != 0:
            #     cv2_img = bridge.imgmsg_to_cv2(self.current_image, "bgr8")
            #     img_filename = 'camera_image_' + str(rospy.get_time()) + '.png'
            #     cv2.imwrite(img_filename, cv2_img) #이미지 저장

            with open(self.filename, 'a') as f:
                data = [trackinfo.my_car.pose, trackinfo.my_car.velocity,
                        trackinfo.dist2obstacles.left, trackinfo.dist2obstacles.middle, trackinfo.dist2obstacles.right,
                        trackinfo.deviation, img_filename] #data로 저장할 list
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(data)

            self.nr_data += 1
            self.my_car_pose = trackinfo.my_car.pose
            self.dist += dist_diff

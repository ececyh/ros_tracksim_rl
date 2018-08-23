#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import rospy

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import math
from math import pi
import numpy as np
import tf
import Trackinfo
import TrackSegment as seg
from geometry_msgs.msg import Twist
import tensorflow as tf



config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared


if __name__ == '__main__':
    rospy.init_node('track_driver', anonymous=True)

    # gp model

    filename = 'gp_5000.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    print 'model loaded : ' + filename

    # NN model
    sess = tf.Session()
    tf.reset_default_graph()
    imported_meta = tf.train.import_meta_graph('nn_epoch-500_Lrate-0.001_neu-256_trackdata_p_fb_degout_obin_deg')
    imported_meta.restore(sess, tf.train.latest_checkpoint('learned_model'))



    ## parameters
    learning_rate = 0.001
    training_epochs = 500
    batch_size = 100
    neu = 256




    listener = tf.TransformListener()

    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform('world', 'vehicle/base_footprint', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        break

    seg.Segment.nr_lane = rospy.get_param("track/nr_lane", 2)
    seg.Segment.lane_width = rospy.get_param("track/lane_width", 4.5)
    seg.Segment.sideline = rospy.get_param("track/sideline", 1.0)

    worldname = rospy.get_param("world", 'straight_2lane')

    Track = []

    if worldname == 'straight_2lane':
        # Straight 2lane          type        origin       length
        Track.append(seg.Segment('straight', (-200, 0, 0), 200))  # 각각의 segment를 생성해 Track에 append
        Track.append(seg.Segment('straight', (0, 0, 0), 200))


    elif worldname == 'curved_2lane':

        # inner radius  = rad - 4.5 - 1
        Track.append(seg.Segment('straight', (-95.13, -101.00, 0), 200))  # 0
        Track.append(seg.Segment('straight', (104.97, -100.76, 0), 100))  # 1
        Track.append(seg.Segment('curved', (204.99, -1.05, -pi / 2), 94.5))  # 2
        Track.append(seg.Segment('straight', (304.96, -1.03, pi / 2), 50))  # 3
        Track.append(seg.Segment('curved', (254.97, 48.94, 0), 44.5))  # 4

        Track.append(seg.Segment('curved', (254.98, 149.59, pi), 44.5, True))  # 5
        Track.append(seg.Segment('curved', (254.98, 149.59, pi / 2), 44.5, True))  # 6
        Track.append(seg.Segment('straight', (254.78, 198.56, 0), 100))  # 7
        Track.append(seg.Segment('curved', (354.73, 298.31, -pi / 2), 94.5))  # 8
        Track.append(seg.Segment('curved', (354.73, 298.31, 0), 94.5))  # 9

        Track.append(seg.Segment('straight', (354.81, 398.29, pi), 200))  # 10
        Track.append(seg.Segment('curved', (154.82, 348.33, pi / 2), 44.5))  # 11
        Track.append(seg.Segment('straight', (104.87, 348.31, -pi / 2), 100))  # 12
        Track.append(seg.Segment('curved', (55.05, 248.42, -pi / 2), 44.5, True))  # 13
        Track.append(seg.Segment('straight', (55.02, 198.57, pi), 100))  # 14

        Track.append(seg.Segment('curved', (-45.19, 148.54, pi / 2), 44.5))  # 15
        Track.append(seg.Segment('curved', (-144.98, 148.57, -pi / 2), 44.5, True))  # 16
        Track.append(seg.Segment('curved', (-144.98, 48.79, pi / 2), 44.5))  # 17
        Track.append(seg.Segment('straight', (-194.96, 48.94, -pi / 2), 50))  # 18
        Track.append(seg.Segment('curved', (-94.99, -1.02, pi), 44.5))  # 19


    elif worldname == 'striaght_4lane':

        Track.append(seg.Segment('straight', (-200, 0, 0), 50))
        Track.append(seg.Segment('straight', (-150, 0, 0), 50))
        Track.append(seg.Segment('straight', (-100, 0, 0), 50))
        Track.append(seg.Segment('straight', (-50, 0, 0), 50))

        Track.append(seg.Segment('straight', (0, 0, 0), 50))
        Track.append(seg.Segment('straight', (50, 0, 0), 50))
        Track.append(seg.Segment('straight', (100, 0, 0), 50))
        Track.append(seg.Segment('straight', (150, 0, 0), 50))

    elif worldname == 'curved_4lane':

        Track.append(seg.Segment('straight', (-118.54, -125.0, 0), 50))
        Track.append(seg.Segment('straight', (-68.72, -125.00, 0), 68.72))
        Track.append(seg.Segment('curved', (0, -66, -pi / 2), 50))
        Track.append(seg.Segment('straight', (58.97, -68.81, pi / 2), 68.81))

        Track.append(seg.Segment('straight', (59.00, 0.00, pi / 2), 50))
        Track.append(seg.Segment('curved', (0, 50, 0), 50))
        Track.append(seg.Segment('straight', (0, 109, pi), 50))
        Track.append(seg.Segment('curved', (-50, 50, pi / 2), 50))

        Track.append(seg.Segment('curved', (-168, 52.04, -pi / 2), 50))
        Track.append(seg.Segment('curved', (-168, -66, pi / 2), 50))
        Track.append(seg.Segment('curved', (-168, -66, pi), 50))
        Track.append(seg.Segment('straight', (-168.32, -124.89, 0), 50))

    elif worldname == 'round_4lane':

        Track.append(seg.Segment('straight', [59, 0, pi / 2], 50))
        Track.append(seg.Segment('curved', [0, 50, 0], 50))
        Track.append(seg.Segment('curved', [0, 50, pi / 2], 50))
        Track.append(seg.Segment('straight', [-59, 50, -pi / 2], 50))
        Track.append(seg.Segment('straight', [-59, 0, -pi / 2], 50))
        Track.append(seg.Segment('curved', [0, -50, -pi], 50))
        Track.append(seg.Segment('curved', [0, -50, -pi / 2], 50))
        Track.append(seg.Segment('straight', [59, -50, pi / 2], 50))

    elif worldname == 'intersection_2lane':

        Track.append(seg.Segment('intersection', [0, 0, 0], 2))
        Track.append(seg.Segment('straight', [-0.03, -57.59, pi / 2], 50))
        Track.append(seg.Segment('straight', [-0.03, 57.49, -pi / 2], 50))
        Track.append(seg.Segment('straight', [-57.53, -0.01, 0], 50))
        Track.append(seg.Segment('straight', [57.47, -0.01, pi], 50))

    elif worldname == 'straight_5lane':

        Track.append(seg.Segment('straight', (-75, 0, 0), 50))
        Track.append(seg.Segment('straight', (-25, 0, 0), 50))
        Track.append(seg.Segment('straight', (25, 0, 0), 50))
        Track.append(seg.Segment('straight', (75, 0, 0), 50))
        Track.append(seg.Segment('straight', (125, 0, 0), 50))
        Track.append(seg.Segment('straight', (175, 0, 0), 50))


    elif worldname == 'straight_6lane':

        Track.append(seg.Segment('straight', (-75, 0, 0), 50))
        Track.append(seg.Segment('straight', (-25, 0, 0), 50))
        Track.append(seg.Segment('straight', (25, 0, 0), 50))
        Track.append(seg.Segment('straight', (75, 0, 0), 50))
        Track.append(seg.Segment('straight', (125, 0, 0), 50))
        Track.append(seg.Segment('straight', (175, 0, 0), 50))

    trackinfo = Trackinfo.Trackinfo()
    trackinfo.set_world(Track)  # 생성한 segment list를 trackinfo에게 알려준다.




    linear_vel = 0.5
    pub = rospy.Publisher('vehicle/cmd_vel', Twist, queue_size=10)


    while True:

        observation = [trackinfo.dist2obstacles.left[0],trackinfo.dist2obstacles.left[1],
                       trackinfo.dist2obstacles.middle[0], trackinfo.dist2obstacles.middle[1],
                       trackinfo.dist2obstacles.right[0], trackinfo.dist2obstacles.right[1],
                       trackinfo.deviation]

        pred = loaded_model.predict(observation)

        msg = Twist()
        msg.linear.x = linear_vel
        msg.angular.z = pred
        pub.publish(msg)

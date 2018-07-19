#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import roslib
import tf
from math import pi
import Trackinfo
import TrackSegment as seg
import Recorder


def printAllInformation(Trackinfo, Recorder): # 그냥 잘 계산하고 있는지 확인하는 함수
    if Trackinfo.my_car.pose:
        print "position : ( x = %.4f, y = %.4f, z = %.4f )" \
              % (Trackinfo.my_car.pose[0], Trackinfo.my_car.pose[1], Trackinfo.my_car.pose[2])
    print "current segment : %d" % Trackinfo.current_segment
    print "lane number : " + str(Trackinfo.lane_number)

    if Trackinfo.leftline: print "left line: [ %s, %s]" % (Trackinfo.leftline[0], Trackinfo.leftline[1])
    if Trackinfo.rightline: print "right line: [ %s, %s]" % (Trackinfo.rightline[0], Trackinfo.rightline[1])

    print "distance : %.4f" % Recorder.dist
    print "number of data: %d \n" % Recorder.nr_data

    print "min dist obstacle"
    print "left : %.4f, %.4f" % (Trackinfo.dist2obstacles.left[0], Trackinfo.dist2obstacles.left[1])
    print "middle : %.4f, %.4f" % (Trackinfo.dist2obstacles.middle[0], Trackinfo.dist2obstacles.middle[1])
    print "right : %.4f, %.4f \n\n" % (Trackinfo.dist2obstacles.right[0], Trackinfo.dist2obstacles.right[1])


if __name__ == '__main__':

    rospy.init_node('track_simulator_main', anonymous=True) # node 시작
    listener = tf.TransformListener()


    rate = rospy.Rate(1) # hz 1초에 한번

    while not rospy.is_shutdown():
        try :
            (trans, rot) = listener.lookupTransform('world','vehicle/base_footprint',rospy.Time(0))
        except (tf.LookupException,tf.ConnectivityException,tf.ExtrapolationException):
            continue

        break



    seg.Segment.nr_lane = rospy.get_param("track/nr_lane", 2)
    seg.Segment.lane_width = rospy.get_param("track/lane_width", 4.5)
    seg.Segment.sideline = rospy.get_param("track/sideline", 1.0)

    worldname = rospy.get_param("world")

    Track = []

    if worldname == 'straight_2lane':
        # Straight 2lane          type        origin       length
        Track.append(seg.Segment('straight', (-200, 0, 0), 200)) # 각각의 segment를 생성해 Track에 append
        Track.append(seg.Segment('straight', (0, 0, 0), 200))


    elif worldname == 'curved_2lane':

        # inner radius  = rad - 4.5 - 1
        Track.append(seg.Segment('straight',(-95.13,-101.00,0),200))
        Track.append(seg.Segment('straight', (104.97,-100.76,0),100))
        Track.append(seg.Segment('curved', (204.99, -1.05,-pi/2),94.5))
        Track.append(seg.Segment('straight', (304.96,-1.03, pi/2),50))
        Track.append(seg.Segment('curved', (254.97,48.94, 0),44.5))

        Track.append(seg.Segment('curved', (254.98,149.59,pi),44.5))#
        Track.append(seg.Segment('curved', (254.98,149.59,pi/2),44.5))#
        Track.append(seg.Segment('straight', (254.78,198.56,0),100))
        Track.append(seg.Segment('curved', (354.73,298.31,-pi/2),94.5))
        Track.append(seg.Segment('curved', (354.73,298.31,0),94.5))

        Track.append(seg.Segment('straight', (354.81,398.29,pi),200))
        Track.append(seg.Segment('curved', (154.82,348.33,pi/2,44.5)))
        Track.append(seg.Segment('straight', (104.87,348.31,-pi/2),100))
        Track.append(seg.Segment('curved', (55.05,248.42,-pi/2),44.5))#
        Track.append(seg.Segment('straight', (55.02,198.57,pi),100))

        Track.append(seg.Segment('curved', (-45.19, 148.54, pi/2), 44.5))
        Track.append(seg.Segment('curved', (-144.98, 148.57, -pi/2),44.5))#
        Track.append(seg.Segment('curved', (-144.98, 48.79, pi/2),44.5))
        Track.append(seg.Segment('straight', (-194.96,48.94, -pi/2),50))
        Track.append(seg.Segment('curved', (-94.99,-1.02,pi),44.5))

    elif worldname == 'striaght_4lane':

        Track.append(seg.Segment('straight', (-200,0,0), 50))
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
        Track.append(seg.Segment('straight',[59,-50,pi/2],50))

    elif worldname == 'intersection_2lane':
        Track.append(seg.Segment('intersection', [0, 0, 0], 2))
        Track.append(seg.Segment('straight', [-0.03,-57.59, pi / 2], 50))
        Track.append(seg.Segment('straight', [-0.03, 57.49, -pi / 2], 50))
        Track.append(seg.Segment('straight', [-57.53, -0.01, 0], 50))
        Track.append(seg.Segment('straight', [57.47, -0.01, pi], 50))

    trackinfo = Trackinfo.Trackinfo()
    recorder = Recorder.Recorder(rospy.get_time())


    trackinfo.set_world(Track) # 생성한 segment list를 trackinfo에게 알려준다.

    while not rospy.is_shutdown():
        printAllInformation(trackinfo, recorder)
        recorder.update(trackinfo) # data 저장해라
        rate.sleep()

# TODO collision detection

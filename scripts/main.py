#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
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
    print "right : %.4f, %.4f" % (Trackinfo.dist2obstacles.right[0], Trackinfo.dist2obstacles.right[1])


if __name__ == '__main__':

    rospy.init_node('track_simulator_main', anonymous=True) # node 시작
    trackinfo = Trackinfo.Trackinfo()
    recorder = Recorder.Recorder(rospy.get_time())

    seg.Segment.nr_lane = rospy.get_param("track/nr_lane", 2)
    seg.Segment.lane_width = rospy.get_param("track/lane_width", 4.5)
    seg.Segment.sideline = rospy.get_param("track/sideline", 1.0)

    Track = []
    # Straight track 4lane
    # Track.append(seg.Segment('straight', (0, -75, PI / 2), 50))
    # Track.append(seg.Segment('straight', (0, -25, PI / 2), 50))
    # Track.append(seg.Segment('straight', (0, 25, PI / 2), 50))

    # Round track 4lane
    # Track.append(seg.Segment('straight', [59, 0, pi / 2], 50))
    # Track.append(seg.Segment('curved', [0, 50, 0], 50))
    # Track.append(seg.Segment('curved', [0, 50, pi / 2], 50))
    # Track.append(seg.Segment('straight', [-59, 50, -pi / 2], 50))
    # Track.append(seg.Segment('straight', [-59, 0, -pi / 2], 50))
    # Track.append(seg.Segment('curved', [0, -50, -pi], 50))
    # Track.append(seg.Segment('curved', [0, -50, -pi / 2], 50))
    # Track.append(seg.Segment('straight',[59,-50,pi/2],50))

    # Straight 2lane          type        origin       length
    Track.append(seg.Segment('straight', (-200, 0, 0), 200)) # 각각의 segment를 생성해 Track에 append
    Track.append(seg.Segment('straight', (0, 0, 0), 200))
    
    trackinfo.set_world(Track) # 생성한 segment list를 trackinfo에게 알려준다.
    
    rate = rospy.Rate(1) # hz 1초에 한번
    while not rospy.is_shutdown():
        printAllInformation(trackinfo, recorder)
        recorder.update(trackinfo) # data 저장해라
        rate.sleep()

# TODO collision detection

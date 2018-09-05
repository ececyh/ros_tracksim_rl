#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import tf
from math import pi
import Trackinfo
import TrackSegment as seg
import Recorder
import argparse
from std_msgs.msg import Int32, Float32
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import Twist

import matplotlib.pyplot as plt

import sys

# state
RUN = 0
RESET = 1
TERMINATE = 2
COLLISION = 3

collision_flag = 0

msg_proxy = {'lane_number':rospy.Publisher("/vehicle/lane_number",Int32, queue_size=10),
'deviation':rospy.Publisher("/vehicle/deviation",Float32, queue_size=10),
'segment_number':rospy.Publisher("/vehicle/lane_number",Int32, queue_size=10)}

def publishMsgInfo(Trackinfo):
    
    msg_proxy['lane_number'].publish(Trackinfo.lane_number)
    msg_proxy['deviation'].publish(Trackinfo.deviation)
    msg_proxy['segment_number'].publish(Trackinfo.current_segment)


def printAllInformation(Trackinfo, Recorder): # 그냥 잘 계산하고 있는지 확인하는 함수

    # position=''
    # global collision_flag
    # if Trackinfo.my_car.pose :
    #     position = "position (x,y,z) : ( %.4f, %.4f, %.4f )"%(Trackinfo.my_car.pose[0], Trackinfo.my_car.pose[1], Trackinfo.my_car.pose[2])
    current_segment = "current segment : %d  "%Trackinfo.current_segment
    lane_number = "lane number : " + str(Trackinfo.lane_number)
    #
    # distance = "distance : %.4f  "%Recorder.dist
    numofdata = "# of data : %d  "%Recorder.nr_data
    #
    # min_dist_obstacle = "min dist obs"
    # frontback = "back     front"
    # left  = "left  : %.4f, %.4f  " % (Trackinfo.dist2obstacles.left[1], Trackinfo.dist2obstacles.left[0])
    # middle =  "middle: %.4f, %.4f" % (Trackinfo.dist2obstacles.middle[1], Trackinfo.dist2obstacles.middle[0])
    # right =  "right : %.4f, %.4f " % (Trackinfo.dist2obstacles.right[1], Trackinfo.dist2obstacles.right[0])
    #
    deviation = "deviation: %.4f" % (Trackinfo.deviation)
    #heading = "heading: %.4f radian %.4f degree" % (Trackinfo.heading, Trackinfo.heading*180/pi)
    steering = "steering: %.4f "%(Trackinfo.steering)

    if mode!='RL':
        # print position
        print current_segment
        print lane_number
        # print distance
        print numofdata + '\n'
        # # print min_dist_obstacle
        # # print frontback
        # # print left
        # # print middle
        # # print right
        print deviation
        #print heading + '\n\n'

        if Trackinfo.deviation*Trackinfo.steering < 0 :
            print steering +'!!\n\n'
        else :
            print steering + '\n\n'



def bumpercallback(data):
    global collision_flag
    if data.states : collision_flag = 1
    else : collision_flag = 0



if __name__ == '__main__':

    mode = 'RL' if 'RL' in sys.argv[1] else 'IL'

    print("--"*30)

    rospy.init_node('track_simulator_main', anonymous=True) # node 시작

    CollisionDetecter = rospy.Subscriber('/vehicle/bumper_state',ContactsState, bumpercallback)

    listener = tf.TransformListener()

    if mode=='RL':
        hz = 120
        print("RL-track-sim environment")
    else:
        print("Imitation Learning")
        hz = 2
    rate = rospy.Rate(hz) # hz 1초에 한번

    # while not rospy.is_shutdown():
    #     try :
    #         (trans, rot) = listener.lookupTransform('world','vehicle/base_footprint',rospy.Time(0))
    #     except (tf.LookupException,tf.ConnectivityException,tf.ExtrapolationException):
    #         continue
    #
    #     break


    seg.Segment.nr_lane = rospy.get_param("track/nr_lane", 2)
    seg.Segment.lane_width = rospy.get_param("track/lane_width", 4.0)
    seg.Segment.sideline = rospy.get_param("track/sideline", 1.0)
    worldname = rospy.get_param("world",'curved_2lane')

    # fig = plt.figure()
    # plt.ion()
    # posplot, = plt.plot(0,0, 'bo')

    Track = []

    if worldname == 'straight_2lane':
        # Straight 2lane          type        origin       length
        Track.append(seg.Segment('straight', (-200, 0, 0), 200)) # 각각의 segment를 생성해 Track에 append
        Track.append(seg.Segment('straight', (0, 0, 0), 200))


    elif worldname == 'curved_2lane':

        # inner radius  = rad - 4.5 - 1
        Track.append(seg.Segment('straight',(-95.13,-101.00,0),200)) #0
        Track.append(seg.Segment('straight', (104.97,-100.76,0),100)) #1
        Track.append(seg.Segment('curved', (204.99, -1.05,-pi/2),100)) #2
        Track.append(seg.Segment('straight', (304.96,-1.03, pi/2),50)) #3
        Track.append(seg.Segment('curved', (254.97,48.94, 0),50)) #4

        Track.append(seg.Segment('curved', (254.986,148.719,pi),50,True)) #5
        Track.append(seg.Segment('curved', (254.983,148.49,pi/2),50,True)) #6
        Track.append(seg.Segment('straight', (254.78,198.56,0),100)) #7
        Track.append(seg.Segment('curved', (354.73,298.31,-pi/2),100)) #8
        Track.append(seg.Segment('curved', (354.73,298.31,0),100)) #9

        Track.append(seg.Segment('straight', (354.81,398.29,pi),200)) #10
        Track.append(seg.Segment('curved', (154.82,348.33,pi/2),50)) #11
        Track.append(seg.Segment('straight', (104.87,348.31,-pi/2),100)) #12
        Track.append(seg.Segment('curved', (55.05,248.42,-pi/2),50,True)) #13
        Track.append(seg.Segment('straight', (55.02,198.57,pi),100)) #14

        Track.append(seg.Segment('curved', (-45.19, 148.54, pi/2), 50)) #15
        Track.append(seg.Segment('curved', (-144.98, 148.57, -pi/2),50,True)) #16
        Track.append(seg.Segment('curved', (-144.98, 48.79, pi/2),50)) #17
        Track.append(seg.Segment('straight', (-194.96,48.94, -pi/2),50)) #18
        Track.append(seg.Segment('curved', (-94.99,-1.02,pi),100)) #19

        # for i in range(len(Track)):
        #     plt.plot(*zip(*Track[i].boundary), lw=3)
        #     for j in range(len(Track[i].lane_pos) / 2):
        #         plt.plot(Track[i].lane_pos[2 * j][0], Track[i].lane_pos[2 * j][1], 'ro')
        #         plt.hold(True)
        #         plt.plot(Track[i].lane_pos[2 * j + 1][0], Track[i].lane_pos[2 * j + 1][1], 'yo')
        #         plt.hold(True)
        #         #plt.show(block=False)
        #
        # plt.hold(True)
        #
        # plt.show(block=False)
        # a = 1


    elif worldname == 'straight_4lane':

        Track.append(seg.Segment('straight', (-200,0,0), 50))
        Track.append(seg.Segment('straight', (-150, 0, 0), 50))
        Track.append(seg.Segment('straight', (-100, 0, 0), 50))
        Track.append(seg.Segment('straight', (-50, 0, 0), 50))

        Track.append(seg.Segment('straight', (0, 0, 0), 50))
        Track.append(seg.Segment('straight', (50, 0, 0), 50))
        Track.append(seg.Segment('straight', (100, 0, 0), 50))
        Track.append(seg.Segment('straight', (150, 0, 0), 50))

    elif worldname == 'straight_4lane_long':

        Track.append(seg.Segment('straight', (-450,0,0), 50))
        Track.append(seg.Segment('straight', (-400,0,0), 50))
        Track.append(seg.Segment('straight', (-350,0,0), 50))
        Track.append(seg.Segment('straight', (-300,0,0), 50))
        Track.append(seg.Segment('straight', (-250,0,0), 50))
        Track.append(seg.Segment('straight', (-200,0,0), 50))
        Track.append(seg.Segment('straight', (-150, 0, 0), 50))
        Track.append(seg.Segment('straight', (-100, 0, 0), 50))
        Track.append(seg.Segment('straight', (-50, 0, 0), 50))

        Track.append(seg.Segment('straight', (0, 0, 0), 50))
        Track.append(seg.Segment('straight', (50, 0, 0), 50))
        Track.append(seg.Segment('straight', (100, 0, 0), 50))
        Track.append(seg.Segment('straight', (150, 0, 0), 50))
        Track.append(seg.Segment('straight', (200, 0, 0), 50))
        Track.append(seg.Segment('straight', (250, 0, 0), 50))
        Track.append(seg.Segment('straight', (300, 0, 0), 50))
        Track.append(seg.Segment('straight', (350, 0, 0), 50))
        Track.append(seg.Segment('straight', (400, 0, 0), 50))

    elif worldname == 'curved_4lane':

        Track.append(seg.Segment('straight', (-150, -100, 0), 50))
        Track.append(seg.Segment('straight', (-100, -100, 0), 50))
        Track.append(seg.Segment('straight', (-50, -100, 0), 50))
        Track.append(seg.Segment('curved', (0, -50, -pi / 2), 50))

        Track.append(seg.Segment('straight', (50, -50, pi / 2), 50))
        Track.append(seg.Segment('straight', (50.00, 0.00, pi / 2), 50))
        Track.append(seg.Segment('curved', (0, 50, 0), 50))
        Track.append(seg.Segment('straight', (0, 100, pi), 50))

        Track.append(seg.Segment('curved', (-50, 50, pi / 2), 50))

        Track.append(seg.Segment('curved', (-150, 50, -pi / 2), 50,True))
        Track.append(seg.Segment('curved', (-150, -50, pi / 2), 50))
        Track.append(seg.Segment('curved', (-150, -50, pi), 50))
        Track.append(seg.Segment('straight', (-150, -100, 0), 50))

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

    elif worldname == 'straight_5lane':

        Track.append(seg.Segment('straight', (0, -75, pi / 2), 50))
        Track.append(seg.Segment('straight', (0, -25, pi / 2), 50))
        Track.append(seg.Segment('straight', (0, 25, pi / 2), 50))
        Track.append(seg.Segment('straight', (0, 75, pi / 2), 50))
        Track.append(seg.Segment('straight', (0, 125, pi / 2), 50))
        Track.append(seg.Segment('straight', (0, 175, pi / 2), 50))


    elif worldname == 'straight_6lane':

        Track.append(seg.Segment('straight', (0, -75, pi/2), 50))
        Track.append(seg.Segment('straight', (0, -25, pi/2), 50))
        Track.append(seg.Segment('straight', (0, 25, pi/2), 50))
        Track.append(seg.Segment('straight', (0, 75, pi/2), 50))
        Track.append(seg.Segment('straight', (0, 125, pi/2), 50))
        Track.append(seg.Segment('straight', (0, 175, pi/2), 50))


    else :
        print('no information about world:' + worldname + '! check out typo ')



    trackinfo = Trackinfo.Trackinfo()
    trackinfo.set_world(Track)  # 생성한 segment list를 trackinfo에게 알려준다.


    recorder = Recorder.Recorder(worldname,hz)
    print 'data will be saved at ' + recorder.directory
    print 'data file name : ' + recorder.filename


    state = RUN
    i = 0

    while True:
        if state == RUN :
            printAllInformation(trackinfo, recorder)
            publishMsgInfo(trackinfo)
            if collision_flag : state = COLLISION
            #recorder.update(trackinfo)  # save data


            # if trackinfo.my_car.pose :
            #     if i is not trackinfo.current_segment :
            #         i = trackinfo.current_segment
            #         plt.clf()
            #         print('clear output')
            #
            #         plt.hold(True)
            #         plt.plot(*zip(*Track[i].boundary), lw = 1)
            #
            #         for j in range(len(Track[i].lane_pos) / 2):
            #             plt.hold(True)
            #             plt.plot(Track[i].lane_pos[2 * j][0], Track[i].lane_pos[2 * j][1], 'ro')
            #             plt.hold(True)
            #             plt.plot(Track[i].lane_pos[2 * j + 1][0], Track[i].lane_pos[2 * j + 1][1], 'yo')
            #
            #         plt.hold(True)
            #         posplot, = plt.plot(trackinfo.my_car.pose[0], trackinfo.my_car.pose[1], 'bo')
            #
            #
            #     plt.hold(True)
            #     posplot.set_data(trackinfo.my_car.pose[0], trackinfo.my_car.pose[1])
            #     fig.canvas.draw()
            #     fig.canvas.flush_events()
            #     plt.show(block=False)


            rate.sleep()




        elif state == COLLISION:

            if mode=='RL':
                state = RUN
            else:
                print "Collision!! Terminate recording."
                state = RESET


        # elif state == RESET:
        #
        #     recorder.reset()
        #     spawner.resetmodels()


        elif state == TERMINATE:

            break





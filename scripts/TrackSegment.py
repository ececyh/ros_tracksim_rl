#!/usr/bin/env python
# -*- coding: utf-8 -*-
# hold information about track and cars
from math import *


class Segment(object):
    numofsegment = 0

    # -------개별 값 ----------------------------
    type = ''          # curved or straight
    segnum = 0         # 몇번째 segment인지
    startpos = tuple   # (x, y, theta), segment의 정중앙 line 시작 pose
    endpos = tuple     # 정중앙 line 끝 pose
    origin = tuple     # curved인 경우 원의 중심, straight인 경우 startpos와 같다
    radius = float     # type = curved 인 경우 원의 반지름
    length = float     # length of segment
    lane_width = 4.0   # width of a lane
    nr_lane = 4        # number of lanes in the segment
    lane_pos = []      # list of lines
    boundary = []      # boundary points
    sideline = 1       #
    reverted = False

    # -----------------------------------------

    def __init__(self, Type, Origin, Len, Rev = False):

        self.segnum = self.numofsegment
        self.numofsegment += 1
        self.type = Type
        self.origin = Origin
        self.reverted = Rev
        theta = 0

        if self.type == 'straight':
            self.length = Len
            self.radius = -1
        elif self.type == 'curved':
            self.length = -1
            self.radius = Len


        # calculate startpos and endpos
        self.lane_pos = []
        if self.type == 'straight':
            self.startpos = Origin
            self.endpos = self.elsum(Origin, (self.length * cos(Origin[2]), self.length * sin(Origin[2]), 0), 3)
            theta = self.startpos[2]

            # append all the line to lane_pos
            # 각 line의 시작점과 끝점을 넣는다.
            b = (self.nr_lane / 2) * self.lane_width
            lane1_start = self.elsum(self.startpos, [-b * sin(theta), b * cos(theta), theta], 3)
            lane1_end = self.elsum(self.endpos, [-b * sin(theta), b * cos(theta), theta], 3)
            self.lane_pos.append(lane1_start)
            self.lane_pos.append(lane1_end)

            i = 1
            while i < self.nr_lane + 1:
                p_lane_start = lane1_start
                p_lane_end = lane1_end

                lane_start = self.elsum(p_lane_start,
                                        (i * self.lane_width * sin(theta), -i * self.lane_width * cos(theta), theta), 3)
                lane_end = self.elsum(p_lane_end,
                                      (i * self.lane_width * sin(theta), -i * self.lane_width * cos(theta), theta), 3)

                self.lane_pos.append(lane_start)
                self.lane_pos.append(lane_end)

                i += 1


        elif self.type == 'curved':
            a = self.radius
            self.startpos = self.elsum(Origin, [a * cos(Origin[2]), a * sin(Origin[2]), pi / 2], 3)
            self.endpos = self.elsum(Origin, [a * cos(Origin[2] + pi / 2), a * sin(Origin[2] + pi / 2), pi], 3)
            theta1 = self.startpos[2]
            theta2 = self.endpos[2]

            b = (self.nr_lane / 2) * self.lane_width
            lane1_start = self.elsum(self.startpos, [-b * sin(theta1), b * cos(theta1), theta1], 3)
            lane1_end = self.elsum(self.endpos, [-b * sin(theta2), b * cos(theta2), theta2], 3)
            self.lane_pos.append(lane1_start)
            self.lane_pos.append(lane1_end)

            i = 1
            while i < self.nr_lane + 1:
                p_lane_start = lane1_start
                p_lane_end = lane1_end

                lane_start = self.elsum(p_lane_start,
                                        (i * self.lane_width * sin(theta1), -i * self.lane_width * cos(theta1), theta1), 3)
                lane_end = self.elsum(p_lane_end,
                                      (i * self.lane_width * sin(theta2), -i * self.lane_width * cos(theta2), theta2), 3)

                self.lane_pos.append(lane_start)
                self.lane_pos.append(lane_end)

                i += 1

            if self.reverted == True :  self.lane_pos.reverse()



            # set boundary
        self.boundary = []
        self.setboundary()

    def elsum(self, a, b, size):
        #elementwise sum.......
        if size == 3:
            return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
        elif size == 2:
            return (a[0] + b[0], a[1] + b[1])

    def setboundary(self):

        if self.type == 'straight':

            # ###############
            #
            # (5)(1)-------------------(2)
            #       -------------------
            #    (4)-------------------(3)
            #
            ###############
                                        # lane1.x               lane1.y
            self.boundary.append((self.lane_pos[0][0], self.lane_pos[0][1]))  # 1
            self.boundary.append((self.lane_pos[1][0], self.lane_pos[1][1]))  # 2

            self.boundary.append((self.lane_pos[2 * (self.nr_lane + 1) - 1][0],
                                  self.lane_pos[2 * (self.nr_lane + 1) - 1][1]))  # 3
            self.boundary.append((self.lane_pos[2 * (self.nr_lane + 1) - 2][0],
                                  self.lane_pos[2 * (self.nr_lane + 1) - 2][1]))  # 4

            self.boundary.append((self.lane_pos[0][0], self.lane_pos[0][1]))  # 5

        elif self.type == 'curved':

            N = 20  # 으로 나누어 점을 찍는다.
            theta = self.origin[2]
            delta = (pi / 2) / N
            i = 0


            # innder boundary - not include sideline
            a = self.radius - (self.nr_lane / 2) * self.lane_width # inner radius
            while i <= N:
                p = self.elsum((self.origin[0], self.origin[1]),
                               (a * cos(theta + delta * i), a * sin(theta + delta * i)), 2)
                self.boundary.append(p)
                i += 1

            i -= 1

            # outer boundary
            b = a + self.nr_lane * self.lane_width #outer radius
            while i >= 0:
                p = self.elsum((self.origin[0], self.origin[1]),
                               (b * cos(theta + delta * i), b * sin(theta + delta * i)), 2)
                self.boundary.append(p)
                i -= 1

            p = self.elsum((self.origin[0], self.origin[1]),
                           (a * cos(theta + delta * 0), a * sin(theta + delta * 0)), 2)
            self.boundary.append(p)

        elif self.type == 'intersection':
            bd_list = [(7.5, 5.5),(5.5, 5.5),(5.5, 7.5),
                       (-5.5, 7.5),(-5.5, 5.5),(-7.5, 5.5),
                       (-7.5, -5.5),(-5.5, -5.5),(-5.5, -7.5),
                       (5.5, -7.5),(5.5, -5.5),(7.5, -5.5),(7.5,5.5)
                       ]

            theta = self.origin[2]

            for p in bd_list :
                rotated_p = (cos(theta)*p[0]-sin(theta)*p[1],sin(theta)*p[0]+cos(theta)*p[1])
                boundary_p = self.elsum((self.origin[0], self.origin[1]), rotated_p, 2)
                self.boundary.append(boundary_p)

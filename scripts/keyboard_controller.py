#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import curses
from geometry_msgs.msg import Twist

screen = curses.initscr()
curses.cbreak()
curses.noecho()
screen.nodelay(True)
screen.keypad(True)

screen.addstr(3, 5, 'Control your car with arrow keys')
screen.addstr(4, 5, '--------------------------------')


def controller():
    linear_vel = 0.0
    angular_vel = 0.0
    linear_step = 0.5
    angular_step = 0.10

    pub = rospy.Publisher('vehicle/cmd_vel', Twist, queue_size=10) # red car : vehicle
    rospy.init_node('controller', anonymous=True)
    rate = rospy.Rate(50)

    while True:
        key = screen.getch()

        if key == curses.KEY_UP:
            screen.addstr(0, 0, 'input: up   ')
            linear_vel += linear_step

        elif key == curses.KEY_DOWN:
            screen.addstr(0, 0, 'input: down ')
            linear_vel -= linear_step

        elif key == curses.KEY_RIGHT:
            screen.addstr(0, 0, 'input: right')
            angular_vel -= angular_step

        elif key == curses.KEY_LEFT:
            screen.addstr(0, 0, 'input: left ')
            angular_vel += angular_step

        elif key == 114:
            linear_vel = 0.0
            angular_vel = 0.0

        screen.addstr(5, 5, 'Linear velocity  : ' + str(linear_vel) + '               ')
        screen.addstr(6, 5, 'Angular velocity : ' + str(angular_vel) + '                 ')
        screen.addstr(7, 5, '--------------------------------')
        screen.addstr(9, 5, 'r : reset every value to 0')

        msg = Twist()
        msg.linear.x = linear_vel
        msg.angular.z = angular_vel
        pub.publish(msg)

        rate.sleep()


if __name__ == '__main__':
    controller()

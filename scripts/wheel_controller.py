#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Joy
from dbw_mkz_msgs.msg import SteeringCmd, BrakeCmd, GearCmd, ThrottleCmd, TurnSignalCmd
from dbw_mkz_msgs.msg import Gear

class wheel_controller():
    def __init__(self):
        self.steering = SteeringCmd()
        self.steering.enable = True
        self.steering.steering_wheel_angle_cmd = 0

        self.accel = ThrottleCmd()
        self.accel.enable = True
        self.accel.pedal_cmd = 0
        self.accel.pedal_cmd_type = 2

        self.brake = BrakeCmd()
        self.brake.enable = True
        self.brake.pedal_cmd = 0
        self.brake.pedal_cmd_type = 2

        self.gear = GearCmd()
        self.gear.cmd.gear = 1

        self.parking_state = 0
        #self.turn_signal = TurnSignalCmd()

        self.pub_steering = rospy.Publisher('vehicle/steering_cmd', SteeringCmd, queue_size=10) # red car : vehicle
        self.pub_brake = rospy.Publisher('vehicle/brake_cmd', BrakeCmd, queue_size=10)
        self.pub_accel = rospy.Publisher('vehicle/throttle_cmd', ThrottleCmd, queue_size=10)
        self.pub_gear = rospy.Publisher('vehicle/gear_cmd', GearCmd, queue_size=10)
        #pub_turn_signal = rospy.Publisher('vehicle/turn_signal_cmd', TurnSignalCmd, queue_size=10)

        rospy.init_node('controller', anonymous=True)
        self.sub = rospy.Subscriber('/joy', Joy, self.callback, queue_size=10)


    def callback(self, data):
        self.steering.steering_wheel_angle_cmd = 4 * data.axes[0]
        self.accel.pedal_cmd = 0.6*(0.5 + 0.5 * data.axes[2])
        self.brake.pedal_cmd = 0.5 + 0.5 * data.axes[3]

        gear_button = data.buttons

        if self.parking_state==0:
            if data.buttons[23]==1:
                self.parking_state = 1
                self.gear.cmd.gear = 0
        elif self.parking_state==1:
            if data.buttons[23]==0:
                self.parking_state = 2
        elif self.parking_state==2 and data.buttons[23]==1:
            self.parking_state = 3
            self.gear.cmd.gear = 1
        elif self.parking_state==3:
            if data.buttons[23]==0:
                self.parking_state = 0        
        elif data.buttons[12] == 1:
            self.gear.cmd.gear = 4  #drive
        elif data.buttons[13] == 1:
            self.gear.cmd.gear = 4
        elif data.buttons[14] == 1:
            self.gear.cmd.gear = 4
        elif data.buttons[15] == 1:
            self.gear.cmd.gear = 4
        elif data.buttons[16] == 1:
            self.gear.cmd.gear = 4
        elif data.buttons[17] == 1:
            self.gear.cmd.gear = 4
        elif data.buttons[18] == 1:
            self.gear.cmd.gear = 2  #reverse
        else:
            self.gear.cmd.gear = 3  #neutral

        print self.accel.pedal_cmd, self.brake.pedal_cmd, self.steering.steering_wheel_angle_cmd

    def control(self):
        rate = rospy.Rate(50)
        while True:
            self.pub_steering.publish(self.steering)
            self.pub_brake.publish(self.brake)
            self.pub_accel.publish(self.accel)
            self.pub_gear.publish(self.gear)

            rate.sleep()

if __name__ == '__main__':
    
    whl = wheel_controller()
    whl.control()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Joy
from dbw_mkz_msgs.msg import SteeringCmd, BrakeCmd, GearCmd, ThrottleCmd, TurnSignalCmd
from dbw_mkz_msgs.msg import Gear

class wheel_controller():
    def __init__(self):
        self.steering = SteeringCmd()
        self.steering.enable = True
        self.steering.steering_wheel_angle_cmd = 0

        self.accel = ThrottleCmd()
        self.accel.enable = True
        self.accel.pedal_cmd = 0
        self.accel.pedal_cmd_type = 2

        self.brake = BrakeCmd()
        self.brake.enable = True
        self.brake.pedal_cmd = 0
        self.brake.pedal_cmd_type = 2

        self.gear = GearCmd()
        self.gear.cmd.gear = 1

        self.parking_state = 0
        #self.turn_signal = TurnSignalCmd()

        self.pub_steering = rospy.Publisher('vehicle/steering_cmd', SteeringCmd, queue_size=10) # red car : vehicle
        self.pub_accel = rospy.Publisher('vehicle/throttle_cmd', ThrottleCmd, queue_size=10)
        self.pub_brake = rospy.Publisher('vehicle/brake_cmd', BrakeCmd, queue_size=10)
        self.pub_gear = rospy.Publisher('vehicle/gear_cmd', GearCmd, queue_size=10)
        #pub_turn_signal = rospy.Publisher('vehicle/turn_signal_cmd', TurnSignalCmd, queue_size=10)

        rospy.init_node('controller', anonymous=True)
        self.sub = rospy.Subscriber('/joy', Joy, self.callback, queue_size=10)


    def callback(self, data):
        self.steering.steering_wheel_angle_cmd = 4 * data.axes[0]
        self.accel.pedal_cmd = 0.6*(0.5 + 0.5 * data.axes[2])
        self.brake.pedal_cmd = 0.5 + 0.5 * data.axes[3]

        gear_button = data.buttons

        if self.parking_state==0:
            if data.buttons[23]==1:
                self.parking_state = 1
                self.gear.cmd.gear = 0
        elif self.parking_state==1:
            if data.buttons[23]==0:
                self.parking_state = 2
        elif self.parking_state==2 and data.buttons[23]==1:
            self.parking_state = 3
            self.gear.cmd.gear = 1
        elif self.parking_state==3:
            if data.buttons[23]==0:
                self.parking_state = 0        
        elif data.buttons[12] == 1:
            self.gear.cmd.gear = 4  #drive
        elif data.buttons[13] == 1:
            self.gear.cmd.gear = 4
        elif data.buttons[14] == 1:
            self.gear.cmd.gear = 4
        elif data.buttons[15] == 1:
            self.gear.cmd.gear = 4
        elif data.buttons[16] == 1:
            self.gear.cmd.gear = 4
        elif data.buttons[17] == 1:
            self.gear.cmd.gear = 4
        elif data.buttons[18] == 1:
            self.gear.cmd.gear = 2  #reverse
        else:
            self.gear.cmd.gear = 3  #neutral

        print self.accel.pedal_cmd, self.brake.pedal_cmd, self.steering.steering_wheel_angle_cmd

    def control(self):
        rate = rospy.Rate(50)
        while True:
            self.pub_steering.publish(self.steering)
            self.pub_brake.publish(self.brake)
            self.pub_accel.publish(self.accel)
            self.pub_gear.publish(self.gear)

            rate.sleep()

if __name__ == '__main__':
    
    whl = wheel_controller()
    whl.control()


#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        # We might need to modify this
        # rfe link: https://discussions.udacity.com/t/solved-stuck-at-steer-value-yawcontroller/499558
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', UnKnown, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.close_waypoint_id = 0

        self.cur_position = None
        self.base_waypoint = None
        self.base_waypoint_header = None
        self.base_waypoint_2d = None
        self.base_waypoint_tree = None
        self.base_waypoint_len = 0
        # styx_msgs/msg/TrafficLight.msg, unknown is 4.
        self.traffic_light = 4

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.cur_position and self.base_waypoint_tree:
                self.close_waypoint_id = self.find_closest_waypoint()
                self.publish()
            rate.sleep()

    def pose_cb(self, msg):
        """ current pose callback
            - update current position
        :param msg:
        """
        self.cur_position = msg.pose.position

    def waypoints_cb(self, msg):
        """ base_waypoints callback
            - load the base_waypoint to class
        :param modified input name: waypoints to msg (avoid confusion)
        """
        self.base_waypoint = msg.waypoints
        self.base_waypoint_header = msg.header
        self.base_waypoint_len = len(msg.waypoints)
        if not self.base_waypoint_2d:
            self.base_waypoint_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in self.base_waypoint]
            self.base_waypoint_tree = KDTree(self.base_waypoint_2d)
        rospy.loginfo(rospy.get_caller_id() + " Total base_waypoint: %s", self.base_waypoint_len)

    def traffic_cb(self, msg):
        # Callback for /traffic_waypoint message. Implement
        self.traffic_light = msg.data

    def obstacle_cb(self, msg):
        # Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def find_closest_waypoint(self):
        x = self.cur_position.x
        y = self.cur_position.y
        closest_id = self.base_waypoint_tree.query([x, y], 1)[1]
        if closest_id > 0:
            # check close_id is in front or back
            cur_close_point = np.array(self.base_waypoint_2d[closest_id])
            prev_close_point = np.array(self.base_waypoint_2d[closest_id - 1])

            vector1 = cur_close_point - prev_close_point
            vector2 = np.array([x, y]) - cur_close_point

            flag = np.dot(vector1, vector2)

            if flag < 0.:
                closest_id = (closest_id + 1) % self.base_waypoint_len
            # rospy.loginfo(rospy.get_caller_id() + " prev_id : %s, curr_id : %s",
            #               self.prev_close_waypoint_id, self.close_waypoint_id)
        return closest_id

    def publish(self):
        lane = Lane()
        # lane.header.frame_id = '/world'
        # lane.header.stamp = rospy.Time(0)
        lane.header = self.base_waypoint_header

        if self.close_waypoint_id + LOOKAHEAD_WPS > self.base_waypoint_len:
            # TODO: we may need to modify the code to deal with out of waypoint probably treat it as a red light or loop
            lane.waypoints = self.base_waypoint[self.close_waypoint_id : self.base_waypoint_len]
            rospy.loginfo(rospy.get_caller_id() + " Close to the end! %s ", self.close_waypoint_id)
        else:
            lane.waypoints = self.base_waypoint[self.close_waypoint_id : self.close_waypoint_id + LOOKAHEAD_WPS]

        self.final_waypoints_pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')

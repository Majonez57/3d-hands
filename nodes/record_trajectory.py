#!/usr/bin/env python3

import rospy
from art_detection.msg import HandResult3D, HandLandmark3D
import numpy
from geometry_msgs.msg import Point
import time
import argparse
import csv 
from collections import defaultdict
import os

trajectories = defaultdict(list)

# When this file is run, hand trajectories are recorded
# and saved as CSV

def points_callback(msg: HandResult3D):
    t = msg.header.stamp.to_sec()

    rospy.loginfo(f"Callback at t:{t}")

    # List of landmarks
    for result3D in msg.landmarks:
        result3D: HandLandmark3D
        x = result3D.point.x
        y = result3D.point.y
        z = result3D.point.z 
        id = result3D.landmarkID

        trajectories[id].append((t, x, y, z))

def save_trajectory():
    path = f"src/art-detection/data/{filename}"
    if not os.path.exists(path):
        os.mkdir(path)

    for landmarkID, trajectory in trajectories.items():
        filepath = f"{path}/landmark_{landmarkID}.csv"
        with open(filepath, 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'y', 'z'])
            for row in trajectory:
                writer.writerow(row)
    

def main():

    input("Press ENTER to begin recording, then use ctrl+C to end.")
    time.sleep(2)
    
    print("RECORDING...")
    
    rospy.init_node('trajectory_recorder')
    sub = rospy.Subscriber("/hand_3d", HandResult3D, points_callback)
    
    rospy.on_shutdown(save_trajectory)
    
    rospy.spin()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "-filename", type=str, required=True)

    args = parser.parse_args()
    filename = args.f
    main()
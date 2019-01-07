#!/usr/bin/env python

# Author: Oscar Argueta
# Date: 05/30/2018
# Kache.ai Inc. Copywrite (c) 2018

"""
Script extracts .jpg images and meta .csv from ROS bags based key inputs from driving logs.
Frames within +- 1 second range are extracted for each key input event. Script ensures only 
unique frames are saved.

Key Map - associates a key press with events of interests

    "i" - incident
    "l" - lane relable - poor lane perfomance
    "s" - situation i.e being pulled over or an event that is out of the ordinary
    "c" - construction
    "v" - vehicle of interest
    " " - merges and exits
    "x" - special case

They output csv is in the following format

    "|bag|timestamp|GPS|ego_speed|key_event|frame|""

Usage:
    python bag_to_images.py --bag_dir=~/Desktop/vide-logs/vidlog-31-05-18-101-280-commute/ \
    --save_dir=../frames2 \
    --skip_rate=2

Flags:
   --bag_dir: path to directory containing ROS bags you wish to process. Must be specified
   --save_dir: path directory you wish to save frames to. Optional. If not specified bar_dir will be used
   --skip_rate: number of frames to skip. Default is value of 1
"""

from __future__ import print_function

import os
import sys
import csv
import glob
import argparse
from tqdm import tqdm

import cv2
import rosbag
from rosbag import ROSBagException
import rospy

from cv_bridge import CvBridge
from keyboard.msg import Key


FIELDS = ["bag", "time_sec", "time_nsec", "GPS", "v_ego", "key_event", "frame"]

KEY_PRESS_LIST = [Key.KEY_c]   

CAR_STATE_TOPIC = "/dbw/toyota_dbw/car_state"
IMAGE_STREAM_TOPIC = "/sensors/usb_cam/rgb/image_raw_f/compressed"


class CSVLogger:
    def __init__(self, path, filename, fields):

        # Create csv file
        file_path = os.path.join(path, filename)
        self.file = open(file_path, 'wb')
        
        # Initialize writer
        self.csv_writer = csv.DictWriter(self.file, fieldnames=fields, delimiter=',')
        self.csv_writer.writeheader()

    def record(self, values_dict):
        self.csv_writer.writerow(values_dict)

    def close(self):
        self.csv_writer.close()


class ImageExporter():
    def __init__(self, FlAGS):
        # Setup extraction directories
        self.bag_dir = FLAGS.bag_dir
        if FLAGS.save_dir:
            self.save_dir = FLAGS.save_dir
        else:
            self.save_dir = FLAGS.bag_dir.rstrip('/') + "_FRAMES"
        print("\nSaving to:", self.save_dir)
    
        self.frames_dir = os.path.join(self.save_dir, "frames")
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        
        # To convert ROS images to OpenCV images so they can be saved.
        self.bridge = CvBridge()
        self.skip_rate = FLAGS.skip_rate

        # Inialize csv logger
        self.csv_logger = CSVLogger(self.save_dir, "frames.csv", FIELDS)
    
    def process_frames(self):
        
        print("\nMining Frames....\o/ \o/ \o/ \o/....... \n")
        rosbag_files = sorted(glob.glob(os.path.join(self.bag_dir, "*.bag")))
        
        # Init extraction loop
        frame_count = 0
        msg_count = 0
        last_key_press = None
        key_press_msgs = []
        frame_type_count = {key: 0 for key in KEY_PRESS_LIST}
        
        # Iterate through bags
        rosbag_files = sorted(glob.glob(os.path.join(self.bag_dir, "*.bag")))
        for bag_file in tqdm(rosbag_files, unit='bag'):  
            # Open bag file. If corrupted skip it
            try:
                with rosbag.Bag(bag_file, 'r') as bag:
                    # Check if desired topics exists in bags
                    recorded_topics = bag.get_type_and_topic_info()[1]
                    if not all(topic in recorded_topics for topic in (CAR_STATE_TOPIC, IMAGE_STREAM_TOPIC)):
                        print("ERROR: Specified topics not in bag file:", bag_file, ".Skipping bag!")
                        continue
                
                    # Get key presses timings
                    for topic, msg, t in bag.read_messages():
                        if topic == CAR_STATE_TOPIC:
                            if msg.keypressed in KEY_PRESS_LIST:
                                if last_key_press is None:
                                    last_key_press = msg
                                elif msg.header.stamp.to_sec() - last_key_press.header.stamp.to_sec() > 0.5:
                                    key_press_msgs.append(msg)
                                    last_key_press = msg

                    
                    # Iterate through Image msgs if there are keypresses of interest
                    if key_press_msgs:
                        print("Extracting Frames from:", bag_file)
                        
                        gps = ""
                        v_ego = 0.0
                        # Extract frames based on key press timings
                        for topic, msg, t in bag.read_messages():
                            # Add frames to buffer for selection
                            if topic == IMAGE_STREAM_TOPIC:
                                for key_press_msg in key_press_msgs:
                                    if abs(msg.header.stamp.to_sec() - key_press_msg.header.stamp.to_sec()) <= 1 \
                                    and msg_count % self.skip_rate == 0:  
                                        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                                        image_name = "frame%04d.jpg" % frame_count
                                        cv2.imwrite(os.path.join(self.frames_dir, image_name), cv_image)
                                        frame_count += 1
                                        
                                        # Update counts
                                        frame_type_count[key_press_msg.keypressed] += 1

                                        # Write to csv
                                        log_dict =dict()
                                        log_dict["bag"] = bag_file.split("/")[-1]
                                        log_dict["time_sec"] = msg.header.stamp.secs
                                        log_dict["time_nsec"] = msg.header.stamp.nsecs
                                        log_dict["GPS"] = gps
                                        log_dict["v_ego"] = v_ego
                                        log_dict["key_event"] = chr(key_press_msg.keypressed)
                                        log_dict["frame"] = image_name
                                        self.csv_logger.record(log_dict)  

                                        # Next frame
                                        break
                                msg_count += 1

                            if topic == CAR_STATE_TOPIC:
                                gps = msg.GPS
                                v_ego = msg.v_ego
                
                msg_count = 0
                last_key_press = None
                key_presses = []
            
            except ROSBagException:
                print("\n",bag_file, "Failed!  || ")
                print(str(ROSBagException.value), '\n')
                continue

        # Print Summary
        print("\nFrames Extracted:", frame_count)
        print("================================")
        [print("Frames from '%s' press:" % chr(key), frame_type_count[key]) for key in KEY_PRESS_LIST]

  
if __name__ == '__main__':
     # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument('-b',"--bag_dir", help="path to bag files")
    parser.add_argument('-s', "--save_dir", help="path to save extracted frames")
    parser.add_argument('-sr', "--skip_rate", type=int, default=1, help="skip every [x] frames. value of 1 skips no frames")
    FLAGS = parser.parse_args()

    # Verify dirs
    if not os.path.exists(FLAGS.bag_dir):
        print("Directory to bag files does not exist", FLAGS.bag_dir)
    elif len(glob.glob(os.path.join(FLAGS.bag_dir, "*.bag"))) < 1 :
	print(os.path.join(FLAGS.bag_dir, "*.bag"))
        print("No bag files in specified directory")
    else:
        image_exporter = ImageExporter(FLAGS)
        image_exporter.process_frames()

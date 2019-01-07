#!/usr/bin/env python

# Author: Dean Webb
# Date: 12/17/2018
# Pronto.ai Inc. Copywrite (c) 2018

"""
Script extracts .jpg images and meta .csv from ROS bags based on perception output from driving logs.
Frames within +- 1 second range are extracted for each key input event. Script ensures only
unique frames are saved.


They output csv is in the following format
    "|bag|timestamp|GPS|ego_speed|class|xmin|ymin|xmax|ymax|confidence|track_id|frame|""

Usage:
    python flicker_detector.py --bag_dir=~/data/kache-workspace/bags/video-logs/vidlog-31-05-18-101-280-commute/ \
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
import json
import ntpath
import csv
import glob
import argparse
import shutil
from tqdm import tqdm
from skimage.feature import hog
from sklearn.preprocessing import RobustScaler, StandardScaler
from collections import OrderedDict
from operator import itemgetter
import copy
from six.moves import cPickle
import pickle
import urllib
from urlparse import urlparse
from datetime import datetime
import pynmea2


import cv2
import numpy as np
import rosbag
from rosbag import ROSBagException
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes
from follow_lead_car.msg import LeadVehicle, TrackerDebug
import rospy

from cv_bridge import CvBridge

# Multiple Object Tracking with Kalman filter
from tracking import deep_sort, sort
from math_utils.fast import check_rectangle_collision



FRAME_FIELDS = ["bag","time_nsec","frame_path",'id','track_id','manual','poly2d','box3d','category','box2d_x1','box2d_y1','box2d_x2','box2d_y2','manualAttributes']
TRACK_FIELDS = ['track_id','frame','hits','age','category','box2d_x1','box2d_y1','box2d_x2','box2d_y2']
CLASSES_LIST = "car,truck,person,construct-post,rider"
DETECTION_TOPIC = "/perception/darknet_ros/bounding_boxes"
IMAGE_STREAM_TOPIC = "/sensors/usb_cam/rgb/image_raw_f/compressed"
CAR_STATE_TOPIC = "/dbw/toyota_dbw/car_state"
# HOG Parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = 8
HOG_CELLS_PER_BLOCK = 2
HOG_CHANNEL = 'ALL' # Can be 0, 1, 2, or "ALL"
SW_SPATIAL_FEAT_FLAG = False
SW_HOG_FEAT_FLAG = False
SW_COLOR_HIST_FEAT_FLAG = True

S3URI = 'kache-scalabel/kache_ai/tracking_frames/'
EXCLUDE_CATS = ['lane', 'drivable area']


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
        if not os.path.exists(os.path.join(self.save_dir, 'bdd100k', 'images')):
            os.makedirs(os.path.join(self.save_dir, 'bdd100k', 'images'), 0o755 )
        self.images_dir = os.path.join(self.save_dir, 'bdd100k', 'images')
        self.max_age = FLAGS.max_age
        self.min_hits = FLAGS.min_hits
        self.max_iou = FLAGS.max_iou
        self.use_cache = FLAGS.use_cache
        self.s3_bucket = FLAGS.s3_bucket
        self.trk_ann_idx = 0
        self.gps = None

        self.trk_table = OrderedDict()
        self.frame_imgs = OrderedDict()
        self.frame_trks = OrderedDict()
        self.flickering_frame_imgs = OrderedDict()
        self.flickering_frame_trks = OrderedDict()



        self.frames_dir = os.path.join(self.save_dir, "frames")
		# Create directories
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)

        # To convert ROS images to OpenCV images so they can be saved.
        self.bridge = CvBridge()
        self.skip_rate = FLAGS.skip_rate

        if type(FLAGS.classes) == str:
		    self.tracking_classes = FLAGS.classes.split(',')
        else:
            self.tracking_classes = FLAGS.classes

		# Initialize Tracker
        self.sort_tracker = deep_sort.Tracker(max_iou_distance=self.max_iou,
                                              max_age=self.max_age,
                                              n_init=self.min_hits)

    def compute_box_center(self, bbox):
        """
        Computes box center from the image top left corner in pixels
        """
        bbox_type = type(bbox)
        if bbox_type == np.ndarray or bbox_type == list or bbox_type == tuple:
            x_coord = bbox[0] + (bbox[2] - bbox[0])/2
            y_coord = bbox[1] + (bbox[3] - bbox[1])/2
        else:
            x_coord = bbox.xmin + (bbox.xmax - bbox.xmin)/2
            y_coord = bbox.ymin + (bbox.ymax - bbox.ymin)/2

        return x_coord, y_coord

    def compute_width_height(self, bbox):
        """
        Computes width and Height of the bounding box in pixels
        """
        if type(bbox) == np.ndarray or type(bbox) == list:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
        else:
            width = bbox.xmax - bbox.xmin
            height = bbox.ymax - bbox.ymin

        return width, height

    def bin_spatial(self, img, size=(64,64)):
        color1 = cv2.resize(img[:,:,0], size).ravel()
        color2 = cv2.resize(img[:,:,1], size).ravel()
        color3 = cv2.resize(img[:,:,2], size).ravel()
        return np.hstack((color1, color2, color3))

    # Define a function to compute color histogram features
    def color_hist(self, img, nbins=128):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        return hist_features

    # Define a function to return HOG features and visualization --
    def get_hog_features(self, img_chan, orient=HOG_ORIENTATIONS,
                         pix_per_cell=HOG_PIXELS_PER_CELL,
                         cell_per_block=HOG_CELLS_PER_BLOCK,
                         vis=False, feature_vec=True):

        features = hog(img_chan, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           visualise=vis)
        return features.ravel()

    def extract_features(self, detection_img):
        """
        Generates a feature vector for the detection
        """

        # Take crop of image around bbox

        # Extract features from bbox

        # Create a list to append feature vectors
        features = []
        cspace = 'YCrCb'
        spatial_size = (64, 64)
        hist_bins = 128

        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(detection_img, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(detection_img, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(detection_img, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(detection_img, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(detection_img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(detection_img)

        # Apply bin_spatial() to get spatial color features
        spatial_features = self.bin_spatial(feature_image, size=spatial_size)

        # Apply color_hist() also with a color space option now
        hist_features = self.color_hist(feature_image, nbins=hist_bins)

        # Call get_hog_features() with vis=False, feature_vec=True
        hog_image = np.copy(cv2.cvtColor(detection_img, cv2.COLOR_RGB2YCrCb))

        hog_shape = np.asarray(hog_image.shape)
        if HOG_CHANNEL == 'ALL':
            hog_features = []
            for channel in range(len(hog_shape)):
                hog_features.append(self.get_hog_features(hog_image[:,:,channel]))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = self.get_hog_features(hog_image[:,:,HOG_CHANNEL])

        # Append the new feature vector to the features list
        # Allow for flagged setting of feature vectors (spatial, hist, hog) must maintain the ordering
        if(SW_SPATIAL_FEAT_FLAG == True and SW_COLOR_HIST_FEAT_FLAG == True and SW_HOG_FEAT_FLAG == True):
            features.append(np.concatenate((spatial_features, hist_features, hog_features)))
        elif(SW_SPATIAL_FEAT_FLAG == False and SW_COLOR_HIST_FEAT_FLAG == True and SW_HOG_FEAT_FLAG == True):
            features.append(np.concatenate((hist_features, hog_features)))
        elif(SW_SPATIAL_FEAT_FLAG == False and SW_COLOR_HIST_FEAT_FLAG == False and SW_HOG_FEAT_FLAG == True):
            features.append(np.array(hog_features))
        elif(SW_SPATIAL_FEAT_FLAG == True and SW_COLOR_HIST_FEAT_FLAG == False and SW_HOG_FEAT_FLAG == True):
            features.append(np.concatenate((spatial_features, hog_features)))
        elif(SW_SPATIAL_FEAT_FLAG == True and SW_COLOR_HIST_FEAT_FLAG == False and SW_HOG_FEAT_FLAG == False):
            features.append(np.array(spatial_features))
        elif(SW_SPATIAL_FEAT_FLAG == True and SW_COLOR_HIST_FEAT_FLAG == True and SW_HOG_FEAT_FLAG == False):
            features.append(np.concatenate((spatial_features, hist_features)))
        elif(SW_SPATIAL_FEAT_FLAG == False and SW_COLOR_HIST_FEAT_FLAG == True and SW_HOG_FEAT_FLAG == False):
            features.append(np.array(hist_features))
        else:
            features.append(np.concatenate(feature_image))

        # Return list of feature vectors
        return np.array(features)



    def tracking_callback(self, msg):
        #======================
        # 1. Get Detections from Topic
        #======================
        detections = []
        for bbox in msg.bounding_boxes:
            duplicate_flag = False
            for det in detections:
                x_bool = bbox.xmin == det[0] and bbox.xmax == det[2]
                y_bool = bbox.ymin == det[1] and bbox.ymax == det[3]
                if x_bool and y_bool:
                    duplicate_flag = True

            # Append only vehicles classified as "car"
            if not duplicate_flag and bbox.Class in self.tracking_classes:
                detections.append([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.probability, bbox.Class])

        # Package detection into deep_sort format
        deep_sort_dets = []
        for det in detections:
            width, height = self.compute_width_height(det)
            bbox_tuple = (det[0], det[1], width, height)

            #======================
            # 1. Calculate features vector for detection
            #======================

            fv = self.extract_features(self.cv_image[det[1]:det[1]+det[3]-1 , det[0]:det[0]+det[2]-1, :]) # Feature Vector
            #print(fv.ravel().tolist())
            # Fit a per-column scaler
            X_scaler = RobustScaler().fit(fv)

            # Apply the scaler to X
            scaled_fv = X_scaler.transform(fv)
            #print(scaled_fv.ravel().tolist())
            d = deep_sort.Detection(bbox_tuple, det[4], scaled_fv.ravel().T, det[5])
            #print("DETECTION:", d.to_tlbr(), '|',  d.confidence)
            deep_sort_dets.append(d)

        #======================
        # 2. Update Tracker
        #======================
        """
        Parameters
        ----------
        mean : ndarray
            Mean vector of the initial state distribution.
        covariance : ndarray
            Covariance matrix of the initial state distribution.
        track_id : int
            A unique track identifier.
        n_init : int
            Number of consecutive detections before the track is confirmed. The
            track state is set to `Deleted` if a miss occurs within the first
            `n_init` frames.
        max_age : int
            The maximum number of consecutive misses before the track state is
            set to `Deleted`.
        feature : Optional[ndarray]
            Feature vector of the detection this track originates from. If not None,
            this feature is added to the `features` cache.

        Attributes
        ----------
        mean : ndarray
            Mean vector of the initial state distribution.
        covariance : ndarray
            Covariance matrix of the initial state distribution.
        track_id : int
            A unique track identifier.
        hits : int
            Total number of measurement updates.
        age : int
            Total number of frames since first occurance.
        time_since_update : int
            Total number of frames since last measurement update.
        state : TrackState
            The current track state.
        features : List[ndarray]
            A cache of features. On each measurement update, the associated feature
            vector is added to this list.
        """


        self.prev_trks = self.sort_tracker.tracks
        self.sort_tracker.update(deep_sort_dets)

        return (deep_sort_dets, self.sort_tracker.tracks)


    def generate_flickering_visualizations(self):
        flickering_frames = [] # (frame_idx, track_id)
        uris2trnpaths = {}
        for track_id in self.trk_table:
            # Get the track with the largest max_age
            oldest_trk_idx, oldest_age = max(enumerate([trk[3] for trk in self.trk_table[track_id]]), key=itemgetter(1))
            # print("OLDEST AGE FOR TRACK", track_id,":", oldest_age)
            for track in self.trk_table[track_id]:
                if track[3] < oldest_age - self.max_age:
                    flickering_frames.append((track[0], track_id))

        # Copy flickering frames from frame_imgs and create s3uris
        for flickering_frame in flickering_frames:
            if flickering_frame[0] not in self.flickering_frame_imgs.keys():
                self.flickering_frame_imgs[flickering_frame[0]] = copy.deepcopy(self.frame_imgs[flickering_frame[0]])

                # Create s3uri
                uri = self.flickering_frame_imgs[flickering_frame[0]]['url']
                uris2trnpaths[uri] = self.load_training_img_uri(uri)
                s3uri = self.generate_s3uri(uris2trnpaths[uri])

                self.flickering_frame_imgs[flickering_frame[0]]['url'] = s3uri
                self.flickering_frame_imgs[flickering_frame[0]]['name'] = uris2trnpaths[uri]

            # Assign track_id as a 'flickering track'
            for trk in self.flickering_frame_imgs[flickering_frame[0]]['labels']:
                if trk['track_id'] == flickering_frame[1] and 'flickering_' not in trk['category']:
                    trk['category'] = 'flickering_'+trk['category']
                    break

            self.flickering_frame_trks[flickering_frame[0]] = copy.deepcopy(self.flickering_frame_imgs[flickering_frame[0]]['labels'])



    def generate_s3uri(self, uri):
        s3uri = os.path.join(self.s3_bucket,self.path_leaf(uri))
        return os.path.join('https://s3-us-west-2.amazonaws.com', s3uri)

    def load_training_img_uri(self, uri):
        trn_uri = os.path.join(self.images_dir, self.path_leaf(uri))
        shutil.copy(uri, trn_uri)
        return trn_uri

    def generate_names_yml(self):
        if not os.path.exists(os.path.join(self.save_dir, 'bdd100k', 'cfg')):
            os.makedirs(os.path.join(self.save_dir, 'bdd100k', 'cfg'), 0o755 )
        self.config_dir = os.path.join(self.save_dir, 'bdd100k', 'cfg')
        self.names_config_yml = os.path.join(self.config_dir, 'flickering_trk_names.yml')

        anns = [i for i in [d for d in [ann for ann in self.flickering_frame_trks.values()]]]
        cats = [[label['category'] for label in labels if label['category'] not in EXCLUDE_CATS] for labels in anns]
        categories = []
        [categories.extend(cat) for cat in cats]
        self.category_names = sorted(set(categories))

        with open(self.names_config_yml, 'w+') as writer:
            for category in sorted(set(self.category_names)):
                writer.write('- name: '+category+'\n')


    def generate_annotations(self):
        ## TODO: Build ground truth in bdd format based on flickers from self.trk_table
        pass

    def export(self, bag_file, force = False, paginate = False):
        if not os.path.exists(os.path.join(self.save_dir, 'bdd100k', 'visualizations')):
            os.makedirs(os.path.join(self.save_dir, 'bdd100k', 'visualizations'), 0o755 )
        if not os.path.exists(os.path.join(self.save_dir, 'bdd100k', 'annotations')):
            os.makedirs(os.path.join(self.save_dir, 'bdd100k', 'annotations'), 0o755 )
        path = os.path.normpath(bag_file)
        self.bdd100k_visualizations = os.path.join(self.save_dir, 'bdd100k/visualizations', "{}-flickering_visualizations.json".format('_'.join(path.split(os.sep))))
        self.bdd100k_annotations = os.path.join(self.save_dir, 'bdd100k/annotations', "{}-flickering_annotations.json".format('_'.join(path.split(os.sep))))

        self.generate_names_yml()

        try:
            os.remove(self.bdd100k_visualizations)
        except OSError: pass

        try:
            os.remove(self.bdd100k_annotations)
        except OSError: pass

        if paginate: # Prepare for Scalabel
            img_data = list(self.flickering_frame_imgs.values())
            for i, chunk in enumerate(self.data_grouper(self.flickering_frame_imgs.values(), 500)):
                tmp =sorted(list(copy.deepcopy(chunk)), key=itemgetter('index'))
                lblidx = 0
                for tmpidx, d in enumerate(tmp):
                    if d: # Reset index
                        tmp[tmpidx]['scalabel_id'] = tmpidx
                        tmp[tmpidx]['videoName'] = "" # Scalabel bug, tmp fix for now
                        tmp[tmpidx]['kache_id'] = int(tmp[tmpidx]['index'])
                        tmp[tmpidx]['index'] = tmpidx
                        # Reset Label ids
                        if d['labels']:
                            for ii, lbl in enumerate(tmp[tmpidx]['labels']):
                                tmp[tmpidx]['labels'][ii]['scalabel_label_id'] = lblidx
                                tmp[tmpidx]['labels'][ii]['kache_label_id'] = int(tmp[tmpidx]['labels'][ii]['id'])
                                tmp[tmpidx]['labels'][ii]['id'] = lblidx
                                lblidx+=1
                data = json.dumps(tmp, indent=4)
                with open('{}_{}.json'.format(os.path.splitext(self.bdd100k_visualizations)[0],i), "w+", encoding='utf-8') as output_json_file:
                    output_json_file.write(data)
        else:
            with open(self.bdd100k_visualizations, "w+") as output_json_file:
                imgs_list = list(self.flickering_frame_imgs.values())
                json.dump(imgs_list, output_json_file)


    def format_from_nanos(self, nanos):
        dt = datetime.fromtimestamp(nanos / 1e9)
        return '{}{:03.0f}'.format(dt.strftime('%Y-%m-%dT%H:%M:%S.%f'), nanos % 1e3)

    def path_leaf(self, path):
        if urlparse(path).scheme != "" or os.path.isabs(path):
            path = os.path.split(path)[-1]
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    def save_cache(self, bag_file):
        path = os.path.normpath(bag_file)
        self._pickle_file = os.path.join(self.save_dir, "{}.pickle".format('_'.join(path.split(os.sep))))
        # Save Data to Pickle
        pickle_dict = {"bagfile": bag_file, "frame_imgs": self.frame_imgs, "trk_table": self.trk_table, "frame_count": self.frame_count}
        print('Saving to Pickle File:', self._pickle_file)
        with open(self._pickle_file,"wb") as pickle_out:
            cPickle.dump(pickle_dict, pickle_out)
        # Inialize CSV Logger / Save Data to CSV
        self.csv_frames_logger = CSVLogger(self.save_dir, path+'FRAMES.csv', FRAME_FIELDS)
        self.csv_trks_logger = CSVLogger(self.save_dir, path+'TRACKS.csv', TRACK_FIELDS)
        self.save_frame_trks_to_csv(bag_file)
        self.save_trk_table_to_csv(bag_file)

    def load_cache(self, bag_file):
        # Initialize cache
        path = os.path.normpath(bag_file)
        self._pickle_file = os.path.join(self.save_dir, "{}.pickle".format('_'.join(path.split(os.sep))))
        if self._pickle_file and os.path.exists(self._pickle_file) and self.use_cache:
            pickle_in = open(self._pickle_file,"rb")
            pickle_dict = cPickle.load(pickle_in)
            return (pickle_dict['frame_imgs'],pickle_dict['trk_table'], pickle_dict['frame_count']+1)
        else: return (None,None, 0)

    def save_trk_table_to_csv(self, bag_file):
        print('Saving to CSV File', self.csv_trks_logger)
        for trk_idx in self.trk_table.keys():
            # (frame, track.to_tlbr(), track.hits, track.age, track.category)
            for missed_track in self.trk_table[trk_idx]:
                log_dict =dict()
                log_dict["track_id"] = trk_idx
                log_dict["frame"] = missed_track[0]
                log_dict["hits"] = missed_track[2]
                log_dict["age"] = missed_track[3]
                log_dict['category'] = missed_track[4]
                log_dict['box2d_x1'] = missed_track[1][0]
                log_dict['box2d_y1'] = missed_track[1][1]
                log_dict['box2d_x2'] = missed_track[1][2]
                log_dict['box2d_y2'] = missed_track[1][3]
                self.csv_logger.record(log_dict)

    def save_frame_trks_to_csv(self, bag_file):
        print('Saving to CSV File', self.csv_frames_logger)
        for frame in self.frame_trks.keys():
            for label in self.frame_trks[frame]:
                log_dict =dict()
                log_dict["bag"] = bag_file.split("/")[-1]
                log_dict["time_nsec"] = frame
                log_dict["frame_path"] = frame_trks[frame]['url']
                log_dict['id'] = label['id']
                log_dict['track_id'] = label['track_id']
                log_dict['manual'] = label['manual']
                log_dict['poly2d'] = label['poly2d']
                log_dict['box3d'] = label['box3d']
                log_dict['category'] = label['category']
                log_dict['box2d_x1'] = label['box2d']['x1']
                log_dict['box2d_y1'] = label['box2d']['y1']
                log_dict['box2d_x2'] = label['box2d']['x2']
                log_dict['box2d_y2'] = label['box2d']['y2']
                log_dict['manualAttributes'] = label['manualAttributes']
                self.csv_logger.record(log_dict)

    def append_frame_trks(self, frame, tracks, dets):
        for det in dets:
            label = {}
            label['id'] = self.trk_ann_idx
            label['track_id'] = None
            self.trk_ann_idx +=1
            label['attributes'] = {'occluded': False,
                                   'truncated': False,
                                   'trafficLightColor': [0, 'NA']}
            label['manual'] = False
            label['poly2d'] = None
            label['box3d'] = None
            label['category'] = 'detected-'+det.category

            det_bb = det.tlwh
            label['box2d'] = {'x1': float(det_bb[0]),
                          'y1':float(det_bb[1]) ,
                          'x2': float(det_bb[0])-1 + float(det_bb[2]) ,
                          'y2': float(det_bb[1])-1 + float(det_bb[3])}
            label['manualAttributes'] = False

            # Append to table
            self.frame_imgs[frame]['labels'].append(label)
            self.frame_trks[frame].append(label)

        for track in tracks:
            label = {}
            label['id'] = self.trk_ann_idx
            label['track_id'] = track.track_id
            self.trk_ann_idx +=1
            label['attributes'] = {'occluded': False,
                                   'truncated': False,
                                   'trafficLightColor': [0, 'NA']}
            label['manual'] = False
            label['poly2d'] = None
            label['box3d'] = None
            label['category'] = 'tracked-'+track.category

            trk_bb = track.to_tlwh()
            label['box2d'] = {'x1': float(trk_bb[0]),
                          'y1':float(trk_bb[1]),
                          'x2': float(trk_bb[0])-1 + float(trk_bb[2]) ,
                          'y2': float(trk_bb[1])-1 + float(trk_bb[3])}
            label['manualAttributes'] = False

            # if label['track_id'] not in set([x['track_id'] for x in self.frame_trks[frame] if label['track_id'] is not None]):
            self.frame_imgs[frame]['labels'].append(label)
            self.frame_trks[frame].append(label)



    def append_trk_table(self, frame, tracks):
        for track in tracks:
            if track.time_since_update > 0: # Append missed frame to self.trk_table
                if track.track_id in self.trk_table and self.trk_table[track.track_id]:
                    self.trk_table[track.track_id].append((frame, track.to_tlbr(), track.hits, track.age, track.category))
                else:
                    self.trk_table[track.track_id] = [(frame, track.to_tlbr(), track.hits, track.age, track.category)]

                print("TRACKING DICT", track.track_id, " UPDATED:",self.trk_table[track.track_id][-1])

    def process_frames(self):
        print("\nMining Frames for Flicker Detection ......\o/ \o/ \o/ \o/....... \n")
        rosbag_files = sorted(glob.glob(os.path.join(self.bag_dir, "*.bag")))
        # Init extraction loop
        self.frame_count = 0
        msg_count = 0

        # Iterate through bags
        rosbag_files = sorted(glob.glob(os.path.join(self.bag_dir, "*.bag")))
        for bag_file in tqdm(rosbag_files, unit='bag'):
            self.frame_imgs, self.trk_table, self.frame_count = self.load_cache(bag_file)

            if not self.frame_imgs or not self.trk_table:
                # Open bag file. If corrupted, skip the file
                try:
                    with rosbag.Bag(bag_file, 'r') as bag:
                        # Check if desired topics exists in bags
                        recorded_topics = bag.get_type_and_topic_info()[1]
                        if not all(topic in recorded_topics for topic in ( DETECTION_TOPIC, CAR_STATE_TOPIC, IMAGE_STREAM_TOPIC)):
                            print("ERROR: Specified topics not in bag file:", bag_file, ".Skipping bag!")
                            continue

                        gps = ""
                        v_ego = 0.0
                        self.cv_image = None
                        # Get Detections
                        time_nsecs = []
                        self.trk_table = OrderedDict()
                        self.frame_imgs = OrderedDict()
                        self.frame_trks = OrderedDict()
                        self.flickering_frame_imgs = OrderedDict()
                        self.flickering_frame_trks = OrderedDict()
                        for topic, msg, t in bag.read_messages():
                            if topic == CAR_STATE_TOPIC:
                                self.gps = msg.GPS
                                self.v_ego = msg.v_ego
                            if topic == IMAGE_STREAM_TOPIC:
                                if msg_count % self.skip_rate == 0:

                                    self.cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                                    image_name = "frame%010d_%s.jpg" % (self.frame_count, str(msg.header.stamp.to_nsec()))
                                    img_path = os.path.join(self.frames_dir, image_name)
                                    cv2.imwrite(img_path, self.cv_image)

                                    uri = img_path
                                    img_key = int(msg.header.stamp.to_nsec())
                                    fname = self.path_leaf(uri)
                                    vid_name = ''
                                    readable_time = self.format_from_nanos(img_key).split(' ')
                                    if ':' in readable_time and ' ' not in readable_time.split(':')[0]:
                                        hour = int(readable_time.split(':')[0])
                                    else:
                                        hour = 12

                                    if (hour > 4 and hour < 6) or (hour > 17 and hour < 19):
                                        timeofday = 'dawn/dusk'
                                    elif hour > 6 and hour < 17:
                                        timeofday = 'daytime'
                                    else:
                                        timeofday = 'night'


                                    scene = 'highway'
                                    timestamp = img_key
                                    dataset_path = img_path
                                    if self.gps:
                                        gps = pynmea2.parse(self.gps)
                                        lat = gps.lat
                                        long = gps.lon
                                    else:
                                        lat = None
                                        long = None
                                    height, width, depth = self.cv_image.shape

                                    self.frame_imgs[img_key] = {'url': img_path,
                                                             'name': img_path,
                                                             'width': width,
                                                             'height': height,
                                                             'index': self.frame_count,
                                                             'timestamp': timestamp,
                                                             'videoName':vid_name,
                                                             'attributes': {'weather': 'clear', 'scene': scene, 'timeofday': timeofday},
                                                             'labels': []
                                                            }
                                    self.frame_trks[img_key] = []

                                    msg_count += 1
                                    self.frame_count += 1

                        ## Get Tracking Data ##
                        for topic, msg, t in bag.read_messages():
                            if topic == DETECTION_TOPIC:
                                # Find corresponding frame for detections message
                                found_frame = False
                                for frame in sorted(self.frame_imgs.keys()):
                                    if int(msg.header.stamp.to_nsec()) > frame-3.333e7 and int(msg.header.stamp.to_nsec()) < frame+3.333e7:
                                        found_frame = True

                                        # Collect tracker_msgs
                                        detections, tracks = self.tracking_callback(msg)
                                        # Append to frame annotations table
                                        self.append_frame_trks(frame, tracks, detections)
                                        # Append to track annotations table
                                        self.append_trk_table(frame, tracks)
                                        # Debugger statement to make monitor data extraction
                                        print("FRAME TIMESTAMP:",frame)
                                        print("DETECTION TIMESTAMP:",int(msg.header.stamp.to_nsec()))
                                        print('IMG PATH:',self.frame_imgs[frame]['url'])
                                        break

                                if not found_frame: # Try a wider time window and try to shuffle detection to appropriate frame
                                    for i, frame in enumerate(sorted(self.frame_imgs.keys())):
                                        if int(msg.header.stamp.to_nsec()) > frame-3.783e7 and int(msg.header.stamp.to_nsec()) < frame+3.783e7:
                                            found_frame = True
                                            # Collect tracks
                                            detections, tracks = self.tracking_callback(msg)
                                            # Append to buffer
                                            if len(self.frame_imgs[frame]) < 2: # Check if already assigned detections to this frame
                                                idx = frame
                                            elif i > 0 and len(self.frame_imgs[self.frame_imgs.keys()[i-1]]) < 2: # Assign detections to the previous frame if empty
                                                idx = self.frame_imgs.keys()[i-1]
                                            elif i < len(self.frame_imgs.keys())-1 and len(self.frame_imgs[self.frame_imgs.keys()[i+1]]) < 2: # Assign detections to the next frame if empty
                                                idx = self.frame_imgs.keys()[i+1]
                                            else:
                                                idx = frame
                                            # Append to frame annotations table
                                            self.append_frame_trks(idx, tracks, detections)
                                            # Append to track annotations table
                                            self.append_trk_table(idx, tracks)
                                            # Debugger statement to make monitor data extraction
                                            print("FRAME TIMESTAMP:",idx)
                                            print("DETECTION TIMESTAMP:",int(msg.header.stamp.to_nsec()))
                                            print('IMG PATH:', self.frame_imgs[idx]['url'])
                                            break
                        self.save_cache(bag_file)
                    msg_count = 0
                except ROSBagException:
                    print("\n",bag_file, "Failed!  || ")
                    print(str(ROSBagException.value), '\n')
                    continue

            ## Generate JSON Object in BDD Format ##
            self.generate_flickering_visualizations()

            ## Generate Ground Truth Annotations in BDD Format ##
            self.generate_annotations()

            # Export to Scalabel
            self.export(bag_file)

            # Print Summary
            print("\nFrames Extracted:", len(self.flickering_frame_imgs.keys()))
            print("=================================================================\n\n")

if __name__ == '__main__':
     # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument('-b',"--bag_dir", help="path to bag files")
    parser.add_argument('-mh',"--min_hits", help="minimum number of detections to be considered a track. Default: 10", default=10)
    parser.add_argument('-mi',"--max_iou", help="maximum iou gating threshold. Default: 0.7", default=0.7)
    parser.add_argument('-ma',"--max_age", help="maximum number of missed detections allowed to continue to be considered a track. Default: 10", default=10)
    parser.add_argument('-s', "--save_dir", help="path to save extracted frames")
    parser.add_argument('-uc',"--use_cache", dest='use_cache', default=True, action='store_true')
    parser.add_argument('-sr', "--skip_rate", type=int, default=1, help="skip every [x] frames. value of 1 skips no frames")
    parser.add_argument('-s3', "--s3_bucket", action="store", help="s3 bucket to lookup and store frames", default=S3URI)
    parser.add_argument('-c', "--classes", action="store", help="tracking classes list. Enter classes as commma-separated values. Default: 'car,truck,person'", default=CLASSES_LIST)
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

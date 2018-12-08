import os
import yaml
import urllib
from PIL import Image
from enum import Enum
from pycocotools.coco import COCO
from ruamel.yaml import YAML

import xml.etree.cElementTree as ET
import glob
import argparse
import numpy as np
import json
import numpy
import cv2
from collections import OrderedDict
import scipy.misc
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon, MultiPoint
import random
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import shutil
import pickle
import pandas as pd
import subprocess
from subprocess import Popen,PIPE,STDOUT,call
from datetime import datetime
import urllib
import subprocess
import pprint
import ntpath
from utils import darknet_annotator
from itertools import zip_longest
import time
from functools import wraps
import http
from pathlib import Path
import copy
import sys
from operator import itemgetter
from collections import defaultdict

class Format(Enum):
    scalabel = 0
    coco = 1
    darknet = 2
    bdd = 3
    vgg = 4
    kache = 5
    open_imgs = 6

DEFAULT_IMG_EXTENSION = '.jpg'
#DEFAULT_IMG_EXTENSION = '.png'
EXCLUDE_CATS = ['lane', 'drivable area']
BASE_DIR = '/media/dean/datastore/datasets/BerkeleyDeepDrive/'
SOURCE_BDD100K_DIR = os.path.join(BASE_DIR, 'bdd100k')
SOURCE_COCO_DIR = '/media/dean/datastore/datasets/road_coco/darknet/data/coco/'
SOURCE_KACHE_DIR =  os.path.join('/media/dean/datastore/datasets/kache_ai', 'frames_dev/kache_set')

## Use old config setup #
# STATIC_NAMES_CONFIG = '/media/dean/datastore/datasets/kache_ai/static_cfg/static.names'
# STATIC_NAMES_CONFIG_YML = '/media/dean/datastore/datasets/kache_ai/static_cfg/static.yml'
# ANNOTATION_MODEL =  "/media/dean/datastore/datasets/darknet/backup/yolov3-bdd100k_51418.weights"
# BASE_DATA_CONFIG = os.path.join('/media/dean/datastore/datasets/darknet', 'cfg', 'bdd100k.data')
# BASE_MODEL_CONFIG = os.path.join('/media/dean/datastore/datasets/darknet', 'cfg', 'yolov3-bdd100k.cfg')
# # Use new config setup #
# ANNOTATION_MODEL =  "/media/dean/datastore/datasets/darknet/detectors/20181111--Testing-4trafficlightcats_1gpu_001lr_64bat_16sd_1020ep_2sb/backup/yolov3-bdd100k_51418.weights"
# BASE_DATA_CONFIG = os.path.join('/media/dean/datastore/datasets/darknet/detectors/20181111--Testing-4trafficlightcats_1gpu_001lr_64bat_16sd_1020ep_2sb/', 'cfg', 'bdd100k.data')
# BASE_MODEL_CONFIG = os.path.join('/media/dean/datastore/datasets/darknet/detectors/20181111--Testing-4trafficlightcats_1gpu_001lr_64bat_16sd_1020ep_2sb/', 'cfg', 'yolov3-bdd100k.cfg')
# STATIC_NAMES_CONFIG = '/media/dean/datastore/datasets/darknet/data/cfg/COCO_train2014_0000.names'
# STATIC_NAMES_CONFIG_YML = '/media/dean/datastore/datasets/darknet/data/cfg/kache_category_names.yml'

# # # Use crude detector config setup #
# ANNOTATION_MODEL =  "/media/dean/datastore/datasets/darknet/detectors/20181201--construction-zones_1gpu_00003lr_32bat_16sd_100ep_4sb/backup/yolov3-bdd100k_68856.weights"
# BASE_DATA_CONFIG = os.path.join('/media/dean/datastore/datasets/darknet/detectors/20181201--construction-zones_1gpu_00003lr_32bat_16sd_100ep_4sb/', 'cfg', 'bdd100k.data')
# BASE_MODEL_CONFIG = os.path.join('/media/dean/datastore/datasets/darknet/detectors/20181201--construction-zones_1gpu_00003lr_32bat_16sd_100ep_4sb/', 'cfg', 'yolov3-bdd100k.cfg')
# STATIC_NAMES_CONFIG = '/media/dean/datastore/datasets/darknet/data/cfg/COCO_train2014_0000.names'
# STATIC_NAMES_CONFIG_YML = '/media/dean/datastore/datasets/darknet/data/cfg/kache_category_names.yml'

ANNOTATION_MODEL =  "/media/dean/datastore/datasets/darknet/detectors/construction-zones_1gpu_0003lr_64bat_16sd_48ep_3sb/backup/yolov3-bdd100k_11000.weights"
BASE_DATA_CONFIG = os.path.join('/media/dean/datastore/datasets/darknet/detectors/construction-zones_1gpu_0003lr_64bat_16sd_48ep_3sb/', 'cfg', 'bdd100k.data')
BASE_MODEL_CONFIG = os.path.join('/media/dean/datastore/datasets/darknet/detectors/construction-zones_1gpu_0003lr_64bat_16sd_48ep_3sb/', 'cfg', 'yolov3-bdd100k.cfg')
STATIC_NAMES_CONFIG = '/media/dean/datastore/datasets/darknet/data/cfg/COCO_train2014_0000.names'
STATIC_NAMES_CONFIG_YML = '/media/dean/datastore/datasets/darknet/data/cfg/kache_category_names.yml'

'''
Current Categories
- name: person
- name: rider
- name: car
- name: truck
- name: bus
- name: train
- name: motor
- name: bike
- name: traffic sign
- name: traffic light
- name: trailer
- name: construct-cone
- name: construct-sign
- name: construct-barrel
- name: construct-pole
- name: construct-equipment
- name: traffic light-red
- name: traffic light-amber
- name: traffic light-green
- name: traffic sign-stop_sign
- name: traffic sign-slow_sign
- name: traffic sign-speed_sign
'''



class DataFormatter(object):
    def __init__(self, annotations_list, s3_bucket = None, check_s3 = False,
                    input_format=Format.scalabel, output_path=os.getcwd(), pickle_file = None,
                    trainer_prefix = None, coco_annotations_file = None, darknet_manifast = None,
                    image_list = None, use_cache = False, use_static_categories = False, convert_attributes = True, combine_categories = True):

        self.input_format = input_format
        self._images = OrderedDict()
        self.trn_anno = OrderedDict()
        self.s3_bucket = s3_bucket
        self.check_s3 = check_s3
        self.output_path = output_path
        self.trainer_prefix = trainer_prefix
        self.use_static_categories = use_static_categories
        os.makedirs(os.path.join(self.output_path, 'coco'), 0o755 , exist_ok = True )
        self.coco_directory = os.path.join(self.output_path, 'coco')
        self.coco_images_dir = os.path.join(self.coco_directory, 'images', self.trainer_prefix.split('_')[1]+'/')
        self.coco_labels_dir = os.path.join(self.coco_directory, 'labels', self.trainer_prefix.split('_')[1]+'/')
        os.makedirs(self.coco_labels_dir, exist_ok = True)
        os.makedirs(self.coco_images_dir, exist_ok = True)
        self.coco_annotations_file = coco_annotations_file

        if darknet_manifast and os.path.exists(darknet_manifast):
            self.darknet_manifast = darknet_manifast
        else:
            self.darknet_manifast = os.path.join(self.coco_labels_dir, 'manifast.txt')

        self.config_dir = os.path.join(os.path.split(self.output_path)[0], 'cfg')
        os.makedirs(self.config_dir, exist_ok = True)

        # Check if pickle_file is None or does not exist\
        path = os.path.normpath(annotations_list)
        self._pickle_file = "{}.pickle".format('_'.join(path.split(os.sep)[5:]))

        if self._pickle_file and os.path.exists(self._pickle_file) and use_cache:
            self._images, self.trn_anno = self.get_cache(self._pickle_file)
        else:
            ###------------------ Kache Logs Data Handler -----------------------###
            if self.input_format == Format.kache:
                # Get images from image_list Directory, check-if annotations_list is pickle file
                if glob.glob(os.path.join(os.path.dirname(image_list), '*.pickle')):
                    pickle_file = glob.glob(os.path.join(os.path.dirname(image_list), '*.pickle'))[0]
                    pickle_in = open(pickle_file,"rb")
                    pickle_dict = pickle.load(pickle_in)
                    self.kache_ai_logs = pickle_dict['img_data']
                    self.kache_ai_lookup_table = pickle_dict['lookup_table']
                    self.input_imgs_dir = os.path.dirname(image_list)
                elif glob.glob(os.path.join(os.path.dirname(annotations_list), '*.pickle')):
                    pickle_file = glob.glob(os.path.join(os.path.dirname(annotations_list), '*.pickle'))[0]
                    pickle_in = open(pickle_file,"rb")
                    pickle_dict = pickle.load(pickle_in)
                    self.kache_ai_logs = pickle_dict['img_data']
                    self.kache_ai_lookup_table = pickle_dict['lookup_table']
                    self.input_imgs_dir = os.path.dirname(annotations_list)
                else:
                    self.input_imgs_dir = os.path.split(annotations_list)[0]

                # imgs_list = glob.glob(os.path.join(self.input_imgs_dir, '*'+DEFAULT_IMG_EXTENSION))
                #
                # uris2paths = {}
                # uris = set([(idx, x) for idx, x in enumerate(imgs_list)])
                # for idx, uri in uris:
                #     if os.path.isfile(uri) and DEFAULT_IMG_EXTENSION in str(uri):
                #         print(uri)


                for bag, frame in ((x1, x2) for x1 in self.kache_ai_logs for x2 in sorted(x1[1]['frames'], key =itemgetter('frame_idx'))):
                    img_data = None

                    imgs_list = glob.glob(os.path.join(self.input_imgs_dir, '*'+DEFAULT_IMG_EXTENSION))

                    uris2paths = {}
                    # uris = set([(idx, x) for idx, x in enumerate(imgs_list)])

                    # for idx, uri in uris:
                    #     if os.path.isfile(uri) and DEFAULT_IMG_EXTENSION in str(uri):
                    #         print(uri)
                    if frame['hash_key'] in [self.path_leaf(uri).replace(DEFAULT_IMG_EXTENSION,'') for uri in imgs_list]:
                        img_data = frame
                        uri = frame['dataset_path']

                        fname = self.path_leaf(uri)
                        img_key, uris2paths[uri] = self.load_training_img_uri(uri)

                        if img_data:
                            if img_data.get('time_readable', None):
                                readable_time = img_data['time_readable'].split(' ')[3]
                            elif img_data.get('time_nsec', None):
                                readable_time = self.format_from_nanos(int(img_data['time_nsec'])).split(' ')[3]
                            else:
                                readable_time = str(datetime.now())

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

                            if img_data.get('bag_name', None):
                                vid_name = img_data['bag_name']
                            else:
                                vid_name = ''

                            scene = 'highway'

                            if img_data.get('time_sec', None):
                                timestamp = img_data['time_sec']
                            else:
                                timestamp = 10000

                            dataset_path = img_data['dataset_path']

                            if img_data.get('latitude', None):
                                lat = img_data['latitude']
                            else:
                                lat = ''

                            if img_data.get('longitude', None):
                                long = img_data['longitude']
                            else:
                                long = ''
                        else:
                            vid_name = None
                            timeofday = 'daytime'
                            scene = 'highway'
                            timestamp = 10000
                            dataset_path = uris2paths[uri]
                            lat = None
                            long = None


                        im = Image.open(uris2paths[uri])
                        width, height = im.size
                        if self.s3_bucket: s3uri = self.send_to_s3(uri)

                        self._images[img_key] = {'url': s3uri,
                                                 'name': self.path_leaf(s3uri),
                                                 'coco_path': dataset_path,
                                                 'width': width,
                                                 'height': height,
                                                 'index': int(frame['frame_idx']),
                                                 'timestamp': int(timestamp),
                                                 'videoName':vid_name,
                                                 'attributes': {'weather': 'clear', 'scene': scene, 'timeofday': timeofday},
                                                 'labels': []
                                                }
                        self.trn_anno[img_key] = []

                self.generate_configs_for_inference()
                print("DATA_CONFIG:", self.current_data_cfg_path)
                print("MODEL_CONFIG:", self.current_model_cfg_path)
                print("MODEL_WEIGHTS:", self.inference_model)
                anns = darknet_annotator.annotate(os.path.abspath(self.input_imgs_dir), self.current_model_cfg_path, self.inference_model, self.current_data_cfg_path)

                ann_idx = 0
                for uri, img_anns in anns:
                    img_key, uris2paths[uri] = self.load_training_img_uri( self.path_leaf(uri) )
                    for ann in img_anns:
                        label = {}
                        label['id'] = int(ann_idx)
                        ann_idx +=1
                        label['attributes'] = {'occluded': False,
                                               'truncated': False,
                                               'trafficLightColor': [0, 'NA']}

                        label['manual'] =  ann.get('manual', False)
                        label['poly2d'] = ann.get('poly2d', None)
                        label['box3d'] = ann.get('box3d', None)
                        label['box2d'] = ann.get('box2d', None)

                        if label['box2d']:
                            assert (label['box2d']['x1'] == ann['box2d']['x1']), "Mismatch: {}--{}".format(label['box2d']['x1'], ann['box2d']['x1'])
                            assert (label['box2d']['x2'] == ann['box2d']['x2']), "Mismatch: {}--{}".format(label['box2d']['x2'], ann['box2d']['x2'])
                            assert (label['box2d']['y1'] == ann['box2d']['y1']), "Mismatch: {}--{}".format(label['box2d']['y1'], ann['box2d']['y1'])
                            assert (label['box2d']['y2'] == ann['box2d']['y2']), "Mismatch: {}--{}".format(label['box2d']['y2'], ann['box2d']['y2'])

                        label['category'] = ann['category']

                        self._images[img_key]['labels'].append(label)

                    self.trn_anno[img_key].extend(self._images[img_key]['labels'])




            ###------------------ Scalabel Data Handler -----------------------###
            if self.input_format == Format.scalabel:
                with open(image_list, 'r') as stream:
                    uris2paths = {}
                    image_data = yaml.load(stream)
                    if image_data:
                        for img in image_data:
                            uri = img['url']

                            fname = os.path.split(uri)[-1]
                            img_key, uris2paths[uri] = self.load_training_img_uri(uri)

                            im = Image.open(uris2paths[uri])
                            width, height = im.size
                            if self.s3_bucket: s3uri = self.send_to_s3(uri)


                            self._images[img_key] = {'url': s3uri, 'name': s3uri, 'coco_path': uris2paths[uri],
                                                              'width': width, 'height': height, 'labels': [],
                                                              'index': int(idx), 'timestamp': 10000,
                                                              'videoName': '',
                                                              'attributes': {'weather': 'clear',
                                                                             'scene': 'undefined',
                                                                             'timeofday': 'daytime'}}
                            self.trn_anno[img_key] = []


                # Import Labels
                with open(annotations_list, 'r') as f:
                    data = json.load(f)

                    for ann in data:
                        fname = os.path.split(ann['url'])[-1]
                        self.trn_anno[img_prefix+fname] = ann['labels']
                        img_data = self._images[img_prefix+fname]
                        img_data['attributes'] = ann['attributes']
                        img_data['videoName'] = ann['videoName']
                        img_data['timestamp'] = ann['timestamp']
                        img_data['index'] = ann['index']

                        self._images[img_prefix+fname] = img_data




            ###------------------ MS COCO Data Handler -----------------------###
            if self.input_format == Format.coco:
                self.coco = COCO(annotations_list)
                with open(annotations_list, 'r') as f:
                    data = json.load(f)


                    annotated_img_idxs = [int(annotation['image_id']) for annotation in data['annotations']]
                    # Add Existing Coco Images
                    imgs = data['images']
                    imgs_list = [(x['id'], x) for x in imgs if int(x['id']) in set(annotated_img_idxs)]

                    uris2paths = {}
                    uris = set([(idx, x['file_name']) for idx, x in imgs_list])
                    ann_idx = 0
                    for idx, uri in uris:
                        fname = os.path.split(uri)[-1]
                        img_key, uris2paths[uri] = self.load_training_img_uri(uri)

                        im = Image.open(uris2paths[uri])
                        width, height = im.size
                        if self.s3_bucket: s3uri = self.send_to_s3(uri)


                        self._images[img_key] = {'url': s3uri, 'name': s3uri, 'coco_path': uris2paths[uri],
                                                          'width': width, 'height': height, 'labels': [],
                                                          'index': idx, 'timestamp': 10000,
                                                          'videoName': '',
                                                          'scalabel_id':idx,'kache_id': idx,
                                                          'attributes': {'weather': 'clear',
                                                                         'scene': None,
                                                                         'timeofday': None}}
                        self.trn_anno[img_key] = []

                        for ann in [l for l in data['annotations'] if int(l['image_id']) == idx]:
                            label = {}
                            label['id'] = ann['id']
                            label['scalabel_label_id'] = int(ann['id'])
                            label['kache_label_id'] = int(ann['id'])
                            label['attributes'] = {'Occluded':False,
                                                   'Truncated': False,
                                                   'Traffic Light Color': [0, 'NA']}

                            label['manual'] = True
                            label['manualAttributes'] = True

                            # Get category name from COCO trainer_prefix
                            cat = self.coco.loadCats([ann['category_id']])[0]

                            if cat and isinstance(cat, list):
                                label['category'] = cat[0]['name']
                            elif cat and isinstance(cat, dict):
                                label['category'] = cat['name']
                            else:
                                label['category'] = None

                            label['box3d'] = None
                            label['poly2d'] = None
                            label['box2d'] = {'x1': "%.3f" % round(float(ann['bbox'][0]),3), 'y1': "%.3f" % round(float(ann['bbox'][1]),3),
                                             'x2':  "%.3f" % round(float(ann['bbox'][0]-1+ann['bbox'][2]),3) , 'y2': "%.3f" % round(float(ann['bbox'][1]-1+ ann['bbox'][3]),3)}
                            self._images[img_key]['labels'].append(label)
                            ann_idx +=1
                        self.trn_anno[img_key].extend(self._images[img_key]['labels'])




            ###------------------ BDD100K Data Handler -----------------------###
            elif self.input_format == Format.bdd:
                BDD100K_VIDEOS_PATH='https://s3-us-west-2.amazonaws.com/kache-scalabel/bdd100k/videos/train/'
                with open(annotations_list, 'r') as f:
                    data = json.load(f)
                    ann_idx = 0
                    for idx, img_label in enumerate(data):
                        img_label_name = img_label['name']
                        if urllib.parse.urlparse(img_label_name).scheme != "" or os.path.isabs(img_label['name']):
                            img_label_name = os.path.split(img_label['name'])[-1]
                        elif not os.path.isabs(img_label['name']):
                            train_type = 'train'
                            if 'val' in self.trainer_prefix and 'train' not in self.trainer_prefix:
                                train_type  = 'val'
                            img_label_name = os.path.join(SOURCE_BDD100K_DIR, 'images/100k', train_type, img_label['name'])

                        img_key, img_uri = self.load_training_img_uri(img_label_name)
                        im = Image.open(img_uri)
                        width, height = im.size
                        if self.s3_bucket: img_uri = self.send_to_s3(img_uri.replace(trainer_prefix,''))

                        # Get GPS Coords
                        lat, long = self.get_gps_coords(img_label_name)


                        if img_label.get('attributes', None):
                            self._images[img_key] = {'url': img_uri, 'name': img_uri, 'coco_path': os.path.join(self.coco_images_dir, self.trainer_prefix.split('_')[1], img_key),
                                                              'width': width, 'height': height, 'labels': [],
                                                              'index': idx, 'timestamp': 10000, 'latitude':lat, 'longitude': long,
                                                              'videoName': BDD100K_VIDEOS_PATH+"{}.mov".format(os.path.splitext(img_label['name'])[0]),
                                                              'attributes': {'weather': img_label['attributes']['weather'],
                                                                             'scene': img_label['attributes']['scene'],
                                                                             'timeofday': img_label['attributes']['timeofday']},
                                                               'scalabel_id':idx,'kache_id': idx}
                        else:
                            self._images[img_key] = {'url': img_uri, 'name': img_uri, 'coco_path': os.path.join(self.coco_images_dir, self.trainer_prefix.split('_')[1], img_key),
                                                              'width': width, 'height': height, 'labels': [],
                                                              'index': idx, 'timestamp': 10000, 'latitude':lat, 'longitude': long,
                                                              'videoName': BDD100K_VIDEOS_PATH+"{}.mov".format(os.path.splitext(img_label['name'])[0]),
                                                              'scalabel_id':idx,'kache_id': idx}

                        self.trn_anno[img_key] = []
                        if img_label.get('labels', None):
                            for ann in [l for l in img_label['labels']]:
                                label = {}
                                label['id'] = int(ann_idx)
                                label['scalabel_label_id'] = int(ann_idx)
                                label['kache_label_id'] = int(ann_idx)
                                label['attributes'] = ann.get('attributes', None)
                                if ann.get('attributes', None):
                                    label['attributes'] = ann['attributes']

                                label['manualShape'] =  ann.get('manualShape', True)
                                label['manualAttributes'] = ann.get('manualAttributes', True)
                                label['poly2d'] = ann.get('poly2d', None)
                                label['box3d'] = ann.get('box3d', None)
                                label['box2d'] = ann.get('box2d', None)

                                if label['box2d']:
                                    assert (label['box2d']['x1'] == ann['box2d']['x1']), "Mismatch: {}--{}".format(label['box2d']['x1'], ann['box2d']['x1'])
                                    assert (label['box2d']['x2'] == ann['box2d']['x2']), "Mismatch: {}--{}".format(label['box2d']['x2'], ann['box2d']['x2'])
                                    assert (label['box2d']['y1'] == ann['box2d']['y1']), "Mismatch: {}--{}".format(label['box2d']['y1'], ann['box2d']['y1'])
                                    assert (label['box2d']['y2'] == ann['box2d']['y2']), "Mismatch: {}--{}".format(label['box2d']['y2'], ann['box2d']['y2'])

                                label['category'] = ann['category']
                                if label['category'] == 'traffic light':
                                    if ann['attributes'].get('trafficLightColor', None):
                                        if ann['attributes']['trafficLightColor'] == 'green':
                                            label['attributes']['Traffic Light Color'] = [1, 'G']
                                        elif ann['attributes']['trafficLightColor'] == 'yellow':
                                            label['attributes']['Traffic Light Color'] = [2, 'Y']
                                        elif ann['attributes']['trafficLightColor'] == 'red':
                                            label['attributes']['Traffic Light Color'] = [3, 'R']
                                    else:
                                        ann['attributes']['Traffic Light Color'] == label['attributes']['Traffic Light Color']
                                self._images[img_key]['labels'].append(label)
                                ann_idx +=1
                            self.trn_anno[img_key].extend(self._images[img_key]['labels'])




            # Convert attributes to Categories
            if convert_attributes:
                self.attributes_to_cats('trafficLightColor')

            if combine_categories:
                self.combine_cats(source_cats = ['bicycle'], target_cat='bike')
                self.combine_cats(source_cats = ['motorcycle'], target_cat='motor')
                self.combine_cats(source_cats = ['stop sign'], target_cat='traffic sign')
                self.combine_cats(source_cats = ['construct-pole', 'construct-cone', 'construct-barrel'], target_cat='construct-post')

            # Save object to picklefile
            pickle_dict = {'images':self._images,'annotations':self.trn_anno}
            print('Saving to Pickle File:', self._pickle_file)
            with open(self._pickle_file,"wb") as pickle_out:
                pickle.dump(pickle_dict, pickle_out)

        print('Length of COCO Images', len(self._images))
        self.show_data_distribution()

    def combine_cats(self, source_cats, target_cat):
        for fname in self._images.keys():
            if self._images[fname].get('labels', None):
                for label in self._images[fname]['labels']:
                    if label['category'] in source_cats:
                        label['category'] = target_cat

                        # Update corresponding annotation
                        for ann in self.trn_anno[fname]:
                            if ann['id'] == label['id']:
                                # overwrite annotation with new label
                                ann = label

    def attributes_to_cats(self, attribute):
        if attribute == 'trafficLightColor':
            for fname in self._images.keys():
                if self._images[fname].get('labels', None):
                    for label in [l for l in  self._images[fname]['labels'] if l['category'] == 'traffic light' and l.get('attributes', None)]:
                        if label['attributes'].get('TrafficLightColor', None):
                            if label['attributes']['trafficLightColor'][1].lower() == 'green' or label['attributes']['trafficLightColor'][1].lower() == 'g'  or label['attributes']['trafficLightColor'][0] == 1:
                                label['category'] = 'traffic light-green'
                            elif label['attributes']['trafficLightColor'][1].lower() == 'yellow' or label['attributes']['trafficLightColor'][1].lower() == 'y'  or label['attributes']['trafficLightColor'][0] == 2:
                                label['category'] = 'traffic light-amber'
                            elif label['attributes']['trafficLightColor'][1].lower() == 'red' or label['attributes']['trafficLightColor'][1].lower() == 'r'  or label['attributes']['trafficLightColor'][0] == 3:
                                label['category'] = 'traffic light-red'
                            else:
                                label['category'] = 'traffic light'

                        # Support both old and new bdd formats
                        elif label['attributes'].get('Traffic Light Color', None):
                            if label['attributes']['Traffic Light Color'][1].lower() == 'g' or label['attributes']['Traffic Light Color'][0] == 1:
                                label['category'] = 'traffic light-green'
                            elif label['attributes']['Traffic Light Color'][1].lower() == 'y' or label['attributes']['Traffic Light Color'][0] == 2:
                                label['category'] = 'traffic light-amber'
                            elif label['attributes']['Traffic Light Color'][1].lower() == 'r' or label['attributes']['Traffic Light Color'][0] == 3:
                                label['category'] = 'traffic light-red'
                            else:
                                label['category'] = 'traffic light'

                        # Update corresponding annotation
                        for ann in self.trn_anno[fname]:
                            if ann['id'] == label['id']:
                                # overwrite annotation with new label
                                ann = label

    def get_gps_coords(self, fname):
        train_type = 'train'
        lat, long = (None, None)
        if 'val' in self.trainer_prefix and 'train' not in self.trainer_prefix:
            train_type  = 'val'

        if self.input_format == Format.bdd and 'bdd100k' in self.s3_bucket:
            finfo = Path(SOURCE_BDD100K_DIR, "info/100k", train_type, "{}.json".format(os.path.splitext(fname)[0]))
            if finfo.exists():
                img_info = json.load(finfo.open())
                if img_info:
                    if img_info.get('gps', None):
                        seq_timestamp = int(img_info['gps'][0]['timestamp'])

                        for gps_info in img_info['gps']:
                            if int(gps_info['timestamp']) == seq_timestamp+10000: # Found frame used in training seq_timestamp
                                lat = float(gps_info['latitude'])
                                long = float(gps_info['longitude'])
                                return (lat,long)
                    elif img_info.get('locations', None):
                        seq_timestamp = int(img_info['locations'][0]['timestamp'])

                        for gps_info in img_info['locations']:
                            if int(gps_info['timestamp']) == seq_timestamp+10000: # Found frame used in training seq_timestamp
                                lat = float(gps_info['latitude'])
                                long = float(gps_info['longitude'])
                                return (lat,long)

        return (lat, long)




    def generate_configs_for_inference(self):
        # Override data config
        self.current_inference_dir = os.path.join(self.output_path, 'inference')
        os.makedirs(os.path.join(self.current_inference_dir, 'data'), exist_ok = True)
        os.makedirs(os.path.join(self.current_inference_dir, 'cfg'), exist_ok = True)
        self.models_path = os.path.abspath(os.path.join(self.current_inference_dir, 'models'))
        os.makedirs(self.models_path, exist_ok = True)
        self.inference_model = os.path.join(self.models_path, self.path_leaf(ANNOTATION_MODEL))
        shutil.copy(ANNOTATION_MODEL, self.inference_model)

        self.generate_names_yml()
        self.generate_names_cfg()

        # Override Data config
        self.current_data_cfg = self.parse_data_config(BASE_DATA_CONFIG)
        self.current_data_cfg = self.inject_data_config(self.current_data_cfg)
        self.current_data_cfg_path = self.save_data_config(self.current_data_cfg, os.path.join(self.current_inference_dir, 'cfg', self.path_leaf(BASE_DATA_CONFIG)))

        # Override model config
        self.current_model_cfg = self.parse_model_config(BASE_MODEL_CONFIG)
        self.current_model_cfg = self.inject_model_config(self.current_model_cfg)
        self.current_model_cfg_path = self.save_model_config(self.current_model_cfg, os.path.join(self.current_inference_dir, 'cfg', self.path_leaf(BASE_MODEL_CONFIG)))

    def parse_model_config(self, path):
        """Parses the yolo-v3 layer configuration file and returns module definitions"""
        file = open(path, 'r')
        lines = file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
        module_defs = []
        for line in lines:
            if line.startswith('['): # This marks the start of a new block
                module_defs.append({})
                module_defs[-1]['type'] = line[1:-1].rstrip()
                if module_defs[-1]['type'] == 'convolutional':
                    module_defs[-1]['batch_normalize'] = 0
            else:
                key, value = line.split("=")
                value = value.strip()
                module_defs[-1][key.rstrip()] = value.strip()

        return module_defs

    def parse_data_config(self, path):
        """Parses the data configuration file"""
        options = dict()
        options['gpus'] = '0,1'
        with open(path, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            key, value = line.split('=')
            options[key.strip()] = value.strip()
        return options


    def save_model_config(self, model_defs, path, overwrite = False):
        if not os.path.exists(path) or overwrite == True:
            """Saves the yolo-v3 layer configuration file"""
            with open(path, 'w') as writer:
                for block in model_defs:
                    writer.write('['+ block['type'] +']'+'\n')
                    [writer.write(str(k)+'='+str(v)+'\n') for k,v in block.items() if k != 'type']
                    writer.write('\n')
        return path


    def save_data_config(self, data_config, path, overwrite = False):
        """Saves the yolo-v3 data configuration file"""
        if not os.path.exists(path) or overwrite == True:
            with open(path, 'w') as writer:
                [writer.write(str(k)+'='+str(v)+'\n') for k,v in data_config.items()]
        return path

    def inject_model_config(self, model_config):
        for i, block in enumerate(model_config):
            if block['type'] == 'yolo':
                block['classes'] = len(self.category_names)
                model_config[i-1]['filters'] = (len(self.category_names)+5)*3
        return model_config


    def inject_data_config(self, data_config):
        data_config['train'] = self.darknet_manifast
        data_config['classes'] = len(self.category_names)
        data_config['valid'] = self.darknet_manifast
        data_config['names'] = self.names_config
        data_config['backup'] = self.models_path
        num_gpus = int(self.parse_nvidia_smi()['Attached GPUs'])
        data_config['gpus'] = ','.join(str(i) for i in range(num_gpus))


        return data_config

    def merge(self, merging_set, include = [], exclude = None, reject_new_categories = True):
        # If any categories in include, merge them with datasets
        include = [x.replace(' ','').lower() for x in include]
        deletions = []
        for fname in merging_set._images.keys():
            delete_marker = True

            for ann in merging_set.trn_anno[fname]:
                ## Only include images with annotations corresponding to this categories in the include array
                if ann['category'].replace('motorcycle', 'motor').replace('bicycle', 'bike').replace('stop sign', 'traffic sign').replace(' ', '').lower() in include:
                    delete_marker = False
                    break

            if delete_marker:
                print('APPENDING to delete list:', fname)
                deletions.append(fname)

        for fname in deletions:
            print('PRUNING the IMAGE deletions ({} total)'.format(len(deletions)))
            merging_set._images.pop(fname)
            merging_set.trn_anno.pop(fname)


        ## Exclude block ##
        deletions = []
        ann_deletions = []

        # Set exclude to all remaining categories if None
        if not exclude:
            exclude = [x for x in merging_set.category_names if x.replace('motorcycle', 'motor').replace('bicycle', 'bike').replace('stop sign', 'traffic sign').replace(' ', '').lower() not in include]
            print('EXCLUDING CATEGORIES:', exclude)
        exclude = [x.replace(' ','').lower() for x in exclude]
        # If any categories in exclude, remove any image associated with categories.
        for fname in merging_set._images.keys():
            # delete_marker = False
            for ann in merging_set.trn_anno[fname]:
                ## Exclude images with annotations corresponding to this category_id
                # if ann['category'].replace(' ', '').lower() not in include:
                #     delete_marker = True
                #     break
                # elif
                if ann['category'].replace('motorcycle', 'motor').replace('bicycle', 'bike').replace('stop sign', 'traffic sign').replace(' ', '').lower() in exclude:
                    ann_deletions.append(ann)

            # if delete_marker:
            #     # Remove images
            #     deletions.append(fname)


        # Prune Images
        for fname in deletions:
            print('PRUNING the IMAGE {} ({} total)'.format(fname, len(deletions)))
            merging_set._images.pop(fname)
            merging_set.trn_anno.pop(fname)

        # Prune annotations
        if reject_new_categories:
            print('PRUNING the ANNOTATION deletions ({} total)'.format(len(ann_deletions)))
            for fname in merging_set._images.keys():
                for ann in ann_deletions:
                    if merging_set.trn_anno.get(fname, None) and ann in merging_set.trn_anno[fname] and ann['category'].replace(' ', '').lower() not in include:
                        merging_set.trn_anno[fname].remove(ann)
                    if merging_set._images.get(fname, None) and ann in merging_set._images[fname]['labels'] and ann['category'].replace(' ', '').lower() not in include:
                        merging_set._images[fname]['labels'].remove(ann)


        # Merge Dataset
        for img_key in merging_set._images.keys():
            self._images[img_key] = copy.deepcopy(merging_set._images[img_key])
            self.trn_anno[img_key] = copy.deepcopy(merging_set.trn_anno[img_key])

        if len( merging_set._images) > 0:
            merge_len = len(merging_set._images)
            merge_ann_len = 0
            for x in merging_set.trn_anno.keys():
                merge_ann_len+=len(merging_set.trn_anno[x])

            print('Successfully merged', merge_len, 'images | and ', merge_ann_len, 'annotations')
        else:
            print('No images left to merge')


        self.export(format = Format.scalabel, paginate = False)
        self.show_data_distribution()

        # Save object to picklefile
        pickle_dict = {'images':self._images,'annotations':self.trn_anno}
        print('Saving to Pickle File:', self._pickle_file)
        with open(self._pickle_file,"wb") as pickle_out:
            pickle.dump(pickle_dict, pickle_out)



    def show_data_distribution(self):
        self.export(format = Format.coco, force = True)
        dataset = {}
        cat_ids = self.coco.getCatIds(catNms=list(self.category_names))

        print('\n'+'#'*11+' DATASET DISTRIBUTION: '+'#'*11+'\n')
        for cat_id in cat_ids:
            annotation_ids = self.coco.getAnnIds(catIds=[cat_id])
            image_ids = self.coco.getImgIds(catIds=[cat_id])
            cat_nm = self.coco.loadCats(ids=[cat_id])[0]['name']
            dataset[cat_id] = (len(annotation_ids), len(image_ids))
            print(cat_nm.upper(), '| Annotations:', dataset[cat_id][0], ' | Images: ',  dataset[cat_id][1])
        print('\n'+'#'*48+'\n')




    def retry(ExceptionToCheck, tries=4, delay=3, backoff=2, logger=None):
        """Retry calling the decorated function using an exponential backoff.

        http://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
        original from: http://wiki.python.org/moin/PythonDecoratorLibrary#Retry

        :param ExceptionToCheck: the exception to check. may be a tuple of
            exceptions to check
        :type ExceptionToCheck: Exception or tuple
        :param tries: number of times to try (not retry) before giving up
        :type tries: int
        :param delay: initial delay between retries in seconds
        :type delay: int
        :param backoff: backoff multiplier e.g. value of 2 will double the delay
            each retry
        :type backoff: int
        :param logger: logger to use. If None, print
        :type logger: logging.Logger instance
        """
        def deco_retry(f):

            @wraps(f)
            def f_retry(*args, **kwargs):
                mtries, mdelay = tries, delay
                while mtries > 1:
                    try:
                        return f(*args, **kwargs)
                    except ExceptionToCheck as e:
                        msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                        if logger:
                            logger.warning(msg)
                        else:
                            print(msg)
                        time.sleep(mdelay)
                        mtries -= 1
                        mdelay *= backoff
                return f(*args, **kwargs)

            return f_retry  # true decorator

        return deco_retry

    @retry(http.client.RemoteDisconnected, tries=5, delay=3, backoff=2)
    @retry(urllib.error.HTTPError, tries=5, delay=3, backoff=2)
    def urlrequest_with_retry(self, source, destination):
        return urllib.request.urlretrieve(source, destination)

    def maybe_download(self, source_uri, destination):
        if not os.path.exists(destination):
            if os.path.exists(source_uri):
                os.makedirs(os.path.split(destination)[0], exist_ok = True)
                shutil.copyfile(source_uri, destination)
            # Try checking coco path for image (since they are mixed)
            elif os.path.exists(os.path.join(SOURCE_COCO_DIR, 'images', self.trainer_prefix.split('_')[1], self.path_leaf(source_uri))):
                source_uri = os.path.join(SOURCE_COCO_DIR, 'images', self.trainer_prefix.split('_')[1], self.path_leaf(source_uri))
                os.makedirs(os.path.split(destination)[0], exist_ok = True)
                shutil.copyfile(source_uri, destination)
            elif os.path.exists(os.path.join(SOURCE_COCO_DIR, 'images', self.trainer_prefix.split('_')[1], self.trainer_prefix+self.path_leaf(source_uri))):
                source_uri = os.path.join(SOURCE_COCO_DIR, 'images', self.trainer_prefix.split('_')[1], self.trainer_prefix+self.path_leaf(source_uri))
                os.makedirs(os.path.split(destination)[0], exist_ok = True)
                shutil.copyfile(source_uri, destination)
            elif urllib.parse.urlparse(source_uri).scheme != "":
                destination, _ = urllib.request.urlretrieve(source_uri, destination)
                statinfo = os.stat(destination)
            elif self.s3_bucket:
                print('SOURCE: ', self.send_to_s3(source_uri.replace(self.trainer_prefix, '')))
                print('DEST: ', destination)
                destination, _ = self.urlrequest_with_retry(self.send_to_s3(source_uri), destination)
            else:
                print('Could not copy file', source_uri, 'to file:', destination, '. Does not exist')


        return destination


    def load_training_img_uri(self, fname):
        train_type = 'train'
        if 'val' in self.trainer_prefix and 'train' not in self.trainer_prefix:
            train_type  = 'val'

        if urllib.parse.urlparse(fname).scheme != "" or os.path.isabs(fname):
            fname = os.path.join(SOURCE_BDD100K_DIR, 'images/100k', train_type, fname)
        else:
            if self.input_format == Format.bdd:
                fname = os.path.join(SOURCE_BDD100K_DIR, 'images/100k', train_type, fname)
            elif self.input_format == Format.coco:
                COCO_DIR =  os.path.join(SOURCE_COCO_DIR, 'images', self.trainer_prefix.split('_')[1])
                fname = os.path.join(COCO_DIR, self.path_leaf(fname))
            elif self.input_format == Format.kache:
                fname = os.path.join(SOURCE_KACHE_DIR, self.path_leaf(fname))

        if self.trainer_prefix not in self.path_leaf(fname):
            img_key = self.trainer_prefix+self.path_leaf(fname)
        else:
            img_key = self.path_leaf(fname)

        ## Add to coco_training_dir
        os.makedirs(os.path.join(self.coco_directory, 'images' , self.trainer_prefix.split('_')[1]), exist_ok = True)
        img_uri = self.maybe_download(fname,
                                    os.path.join(self.coco_directory, 'images' , self.trainer_prefix.split('_')[1], img_key))
        return img_key, img_uri

    def get_cache(self, pickle_file):
        self._pickle_file = pickle_file
        pickle_in = open(self._pickle_file,"rb")
        pickle_dict = pickle.load(pickle_in)
        return (pickle_dict['images'],pickle_dict['annotations'])

    def send_to_s3(self, img_path):
        s3_path = os.path.join(self.s3_bucket,self.path_leaf(img_path))

        if self.check_s3:
            #exists = subprocess.check_output("aws s3 ls {}".format(s3_path), shell=True)
            # if not exists:
            s3_bucket = 's3://'+self.s3_bucket
            sp = subprocess.Popen("aws s3 cp {} {}".format(img_path, s3_bucket), shell=True, stdout=PIPE)
            out_str = sp.communicate()
            print(out_str[0].decode("utf-8"))
        return os.path.join('https://s3-us-west-2.amazonaws.com', s3_path)

    def download_from_s3(self, img_path):
        if self.input_format == Format.bdd:
            uri = img_path.replace(self.trainer_prefix,'')
        else:
            uri = img_path
        s3uri = self.send_to_s3(uri)
        res = os.system("curl -o {} {}".format(img_path, s3uri))
        return img_path

    def generate_names_cfg(self):
        self.names_config = os.path.join(self.config_dir, self.trainer_prefix+'.names')
        if self.use_static_categories:
            with open(STATIC_NAMES_CONFIG, "r") as reader:
                cats = [x.strip('\n') for x in reader]
                self.category_names =  cats

            with open(self.names_config, 'w+') as writer:
                for category in self.category_names:
                    writer.write(category+'\n')
        else:
            with open(self.names_config, 'w+') as writer:
                for category in sorted(set(self.category_names)):
                    writer.write(category+'\n')




    def collect_config(self):
        yaml = YAML()
        with open(STATIC_NAMES_CONFIG_YML, "r") as reader:
            code = yaml.load(reader)
            print(type(code))
            yaml.dump(codge, sys.stdout)


    def generate_names_yml(self):
        self.names_config_yml = os.path.join(self.config_dir, self.trainer_prefix+'_names.yml')
        if self.use_static_categories:
            with open(STATIC_NAMES_CONFIG_YML, "r") as reader:
                cats = [x.strip('\n').replace('- name: ', '') for x in reader]
                self.category_names =  cats

            with open(self.names_config_yml, 'w+') as writer:
                for category in self.category_names:
                    writer.write('- name: '+category+'\n')
        else:
            anns = [i for i in [d for d in [ann for ann in self.trn_anno.values()]]]
            cats = [[label['category'] for label in labels if label['category'] not in EXCLUDE_CATS] for labels in anns]
            categories = []
            [categories.extend(cat) for cat in cats]
            self.category_names = sorted(set(categories))

            with open(self.names_config_yml, 'w+') as writer:
                for category in sorted(set(self.category_names)):
                    writer.write('- name: '+category+'\n')



    def format_from_nanos(self, nanos):
        dt = datetime.datetime.fromtimestamp(nanos / 1e9)
        return '{}{:03.0f}'.format(dt.strftime('%Y-%m-%dT%H:%M:%S.%f'), nanos % 1e3)


    def path_leaf(self, path):
        if urllib.parse.urlparse(path).scheme != "" or os.path.isabs(path):
            path = os.path.split(path)[-1]

        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)


    def convert_anns_to_coco(self):
        images, anns = [], []
        start_idx, ann_index = int(1e7), int(1e7)
        self.num_imgs = len(self.trn_anno.keys())

        for img_id, fname in enumerate(self.trn_anno.keys(), start=start_idx):
            width, height = self._images[fname]['width'], self._images[fname]['height']
            fname = self.path_leaf(fname)
            if not fname.startswith(self.trainer_prefix):
                fname = self.trainer_prefix+fname
            dic = {'file_name': fname, 'id': img_id, 'height': height, 'width': width}
            images.append(dic)

            for annotation in [x for x in self.trn_anno[fname] if x['category'] in self.category_names and x['box2d']]:
                bb = annotation['box2d']

                if bb:
                    xstart, ystart, xstop, ystop = float(bb['x1']),float(bb['y1']),float(bb['x2']),float(bb['y2'])

                    if xstart < 0: xstart = 0.0
                    if ystart < 0: ystart = 0.0
                    if ystop <= 0: ystop = 3.0
                    if xstop <= 0: xstop = 3.0

                    # Get Points from Bounding Box
                    pts = []
                    pts.append((xstart , xstop))
                    pts.append((xstop , ystart))
                    pts.append((xstop , ystop))
                    pts.append((xstart , ystop))

                    segmentations = []
                    segmentations.append([])
                    width = xstop - xstart
                    height = ystop - ystart
                    bb = (xstart, ystart, width, height)
                    area = float(width*height)

                    annotation = {
                        'segmentation': segmentations,
                        'iscrowd': 0,
                        'image_id': img_id,
                        'category_id': self.cats2ids[annotation['category']],
                        'id': ann_index,
                        'bbox': bb,
                        'area': area
                    }
                    ann_index+=1
                    anns.append(annotation)
        return anns, images

    def generate_coco_annotations(self):
        if self.use_static_categories:
            with open(STATIC_NAMES_CONFIG, "r") as reader:
                cats = [x.strip('\n') for x in reader]
                self.category_names =  set(cats)
        else:
            anns = [i for i in [d for d in [ann for ann in self.trn_anno.values()]]]
            cats = [[label['category'] for label in labels if label['category'] not in EXCLUDE_CATS] for labels in anns]
            categories = []
            [categories.extend(cat) for cat in cats]
            self.category_names = sorted(set(categories))
        self.cats2ids, self.ids2cats = {}, {}

        for i, label in enumerate(self.category_names):
            self.cats2ids[str(label).lower()] = i
        self.ids2cats = {i: v for v, i in self.cats2ids.items()}

        self.coco_categories = []
        for c in self.category_names:
            self.coco_categories.append({"id": self.cats2ids[c], "name": c, "supercategory":c})


        coco_anns, coco_imgs = self.convert_anns_to_coco()
        print('Length of COCO Annotations:', len(coco_anns))



        INFO = {
            "description": "Road Object-Detections Dataset based on MS COCO",
            "url": "https://kache.ai",
            "version": "0.0.1",
            "year": 2018,
            "contributor": "deanwebb",
            "date_created": datetime.utcnow().isoformat(' ')
        }

        LICENSES = [
            {
                "id": 1,
                "name": "The MIT License (MIT)",
                "url": "https://opensource.org/licenses/MIT",
                "description":  """
                                The MIT License (MIT)
                                Copyright (c) 2017 Matterport, Inc.

                                Permission is hereby granted, free of charge, to any person obtaining a copy
                                of this software and associated documentation files (the "Software"), to deal
                                in the Software without restriction, including without limitation the rights
                                to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
                                copies of the Software, and to permit persons to whom the Software is
                                furnished to do so, subject to the following conditions:

                                The above copyright notice and this permission notice shall be included in
                                all copies or substantial portions of the Software.

                                THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
                                IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
                                FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
                                AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
                                LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
                                OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
                                THE SOFTWARE.
                                """
            }
        ]

        coco_output = {'info': INFO, 'licenses': LICENSES, 'images': coco_imgs, 'annotations': coco_anns, 'categories': self.coco_categories}
        os.makedirs(os.path.join(self.coco_directory, 'annotations'), exist_ok = True)
        self.coco_annotations_file = os.path.join(self.coco_directory, 'annotations', '{}_annotations.json'.format(self.trainer_prefix))
        with open(self.coco_annotations_file, 'w+') as output_json_file:
            json.dump(coco_output, output_json_file)


    def data_grouper(self, iterable, n, fillvalue={}):
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)

    def parse_nvidia_smi(self):
        sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str = sp.communicate()
        out_list = out_str[0].decode("utf-8").split('\n')
        out_dict = {}

        for item in out_list:
            try:
                key, val = item.split(':')
                key, val = key.strip(), val.strip()
                out_dict[key] = val
            except:
                pass

        return out_dict


    def convert_coco_to_yolo(self):
        darknet_conversion_results = os.path.join(self.coco_labels_dir,'convert2yolo_results.txt')
        par_path = os.path.abspath(os.path.join(self.output_path, os.pardir, os.pardir, os.pardir))
        val_par_path = os.path.abspath(os.path.join(par_path, os.pardir))

        if 'darknet' in os.path.split(os.path.abspath(self.output_path))[1].strip('/'):
            yolo_converter = os.path.join(os.path.abspath(par_path), 'convert2Yolo/example.py')
        elif 'darknet' in os.path.split(os.path.abspath(os.path.join(self.output_path, os.pardir, os.pardir, os.pardir, os.pardir)))[1].strip('/'):
            yolo_converter = os.path.join(os.path.abspath(val_par_path), 'convert2Yolo/example.py')
        else:
            # yolo_converter = os.path.join(os.path.abspath(par_path),'darknet', 'convert2Yolo/example.py')
            yolo_converter = os.path.join('/media/dean/datastore/datasets/darknet', 'convert2Yolo/example.py')

        os.makedirs(os.path.abspath(os.path.join(darknet_conversion_results, os.pardir)), exist_ok = True)
        if not os.path.exists(darknet_conversion_results):
            coco2yolo = "python3 {} --datasets COCO --img_path \"{}\" --label \"{}\" --convert_output_path \"{}\" --img_type \"{}\" --manipast_path {} --cls_list_file {} | tee -a  {}".format(
                                yolo_converter, self.coco_images_dir, self.coco_annotations_file,
                                self.coco_labels_dir, DEFAULT_IMG_EXTENSION, os.path.split(self.darknet_manifast)[0], self.names_config,
                                darknet_conversion_results)

            print('\nConverting annotations into Darknet format. Directory:',self.coco_labels_dir)
            print('\nCoco to Yolo command:', coco2yolo)
            res = os.system(coco2yolo)


    def export(self, format = Format.coco, force = False, paginate = True):
        if format == Format.coco:
            if not self.coco_annotations_file or not os.path.exists(self.coco_annotations_file) or force == True:
                self.generate_coco_annotations()
                self.generate_names_cfg()
                self.coco = COCO(self.coco_annotations_file)

        elif format == Format.darknet:
            if not self.coco_annotations_file or not os.path.exists(self.coco_annotations_file) or force == True:
                # Convert to COCO first, since Darknet expects it
                self.export(format = Format.coco, force = force)

                if not self.darknet_manifast or not os.path.exists(self.darknet_manifast)  or force == True:
                    if not self.coco_labels_dir or not os.path.exists(self.coco_labels_dir):
                        self.coco_labels_dir = os.path.join(self.coco_directory, 'labels', self.trainer_prefix.split('_')[1]+'/')
                        os.makedirs(self.coco_labels_dir, exist_ok = True)

                    if not self.darknet_manifast or not os.path.exists(self.darknet_manifast):
                        self.darknet_manifast = os.path.join(self.coco_labels_dir, 'manifast.txt')
                    self.convert_coco_to_yolo()

        elif format == Format.scalabel:
            os.makedirs(os.path.join(self.output_path, 'bdd100k', 'annotations'), 0o755 , exist_ok = True )
            self.bdd100k_annotations = os.path.join(self.output_path, 'bdd100k', 'annotations/bdd100k_altered_annotations.json')
            self.generate_names_yml()

            try:
                os.remove(self.bdd100k_annotations)
            except OSError: pass

            if paginate: # Prepare for Scalabel
                img_data = list(self._images.values())
                for i, chunk in enumerate(self.data_grouper(self._images.values(), 1000)):
                    tmp =sorted(list(copy.deepcopy(chunk)), key=itemgetter('index'))
                    lblidx = 0
                    for tmpidx, d in enumerate(tmp):
                        if d: # Reset index
                            tmp[tmpidx]['scalabel_id'] = tmpidx
                            tmp[tmpidx]['videoName'] = "" # Scalabel bug, tmp fix for noew
                            tmp[tmpidx]['kache_id'] = int(tmp[tmpidx]['index'])
                            tmp[tmpidx]['index'] = tmpidx
                            # Reset Label ids
                            if d['labels']:
                                #tmp_lst = [x for x in d['labels'] if not x['category'] in EXCLUDE_CATS]
                                #tmp[tmpidx]['labels'] = tmp_lst
                                for ii, lbl in enumerate(tmp[tmpidx]['labels']):
                                    tmp[tmpidx]['labels'][ii]['scalabel_label_id'] = lblidx
                                    tmp[tmpidx]['labels'][ii]['kache_label_id'] = int(tmp[tmpidx]['labels'][ii]['id'])
                                    tmp[tmpidx]['labels'][ii]['id'] = lblidx
                                    lblidx+=1


                    data = json.dumps(tmp, indent=4)
                    with open('{}_{}.json'.format(os.path.splitext(self.bdd100k_annotations)[0],i), "w+", encoding='utf-8') as output_json_file:
                        output_json_file.write(data)
            else:
                with open(self.bdd100k_annotations, "w+") as output_json_file:
                    imgs_list = list(self._images.values())
                    json.dump(imgs_list, output_json_file)

        elif format == Format.bdd:
                    os.makedirs(os.path.join(self.output_path, 'bdd100k', 'annotations'), 0o755 , exist_ok = True )
                    self.bdd100k_annotations = os.path.join(self.output_path, 'bdd100k', 'annotations/bdd100k_altered_annotations.json')
                    self.generate_names_yml()

                    try:
                        os.remove(self.bdd100k_annotations)
                    except OSError: pass

                    if paginate:
                        img_data = list(self._images.values())
                        for i, chunk in enumerate(self.data_grouper(self._images.values(), 1000)):
                            with open('{}_{}.json'.format(os.path.splitext(self.bdd100k_annotations)[0],i), "w+") as output_json_file:
                                json.dump(list(chunk), output_json_file)
                    else:
                        with open(self.bdd100k_annotations, "w+") as output_json_file:
                            imgs_list = list(self._images.values())
                            json.dump(imgs_list, output_json_file)

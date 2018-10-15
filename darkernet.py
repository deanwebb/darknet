#!/usr/bin/env python


import argparse
import os
import yaml
import urllib
from PIL import Image
from enum import Enum
from pycocotools.coco import COCO
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
import random
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import shutil
import pickle
import pandas as pd
from subprocess import Popen,PIPE,STDOUT,call
from utils import datasets


## Configuration Parameters ##
BDD100K_DIR = os.path.join('/media/dean/datastore1/datasets/BerkeleyDeepDrive', 'bdd100k')
WORKING_DIR = os.getcwd()
TRAINERS_DIR = os.path.join(WORKING_DIR, 'trainers')
ANNOTATIONS_LIST = os.path.join(BDD100K_DIR, 'labels/bdd100k_labels_images_train.json')
COCO_ANNOTATIONS_LIST = os.path.join('/media/dean/datastore1/datasets/road_coco/darknet/data/coco/annotations/instances_train2014.json')
BASE_DATA_CONFIG = os.path.join(WORKING_DIR, 'cfg', 'bdd100k.data')
BASE_MODEL_CONFIG = os.path.join(WORKING_DIR, 'cfg', 'yolov3-bdd100k.cfg')
S3_BUCKET = 'kache-scalabel/coco_dataset/images/train2014/'


class Darkernet():
    def __init__(self, ARGS):
        if (ARGS.current_working_dir):
            self.current_working_dir = ARGS.current_working_dir
        else:
            self.current_working_dir = os.getcwd()
        if (ARGS.trainers_dir):
            self.trainers_dir = ARGS.trainers_dir
        else:
            self.trainers_dir = os.path.join(self.current_working_dir, 'trainers')
        self.trainers = os.listdir(self.trainers_dir)

        if (ARGS.annotations_list):
            self.annotations_list = ARGS.annotations_list
        if (ARGS.model_cfg):
            self.model_cfg = ARGS.model_cfg
        if (ARGS.data_cfg):
            self.data_cfg = ARGS.data_cfg
        if (ARGS.s3_bucket):
            self.s3_bucket = ARGS.s3_bucket
        if (ARGS.weights):
            self.current_weights = ARGS.weights
        if (ARGS.resume):
            self.resume = True
        else:
            self.resume = False
        if (ARGS.format):
            if (ARGS.format == 'BDD'):
                self.input_format = datasets.Format.bdd
            elif (ARGS.format == 'COCO'):
                self.input_format = datasets.Format.coco
            elif (ARGS.format == 'OPENIMGS'):
                self.input_format = datasets.Format.open_imgs
            elif (ARGS.format == 'SCALABEL'):
                self.input_format = datasets.Format.scalabel
            elif (ARGS.format == 'VGG'):
                self.input_format = datasets.Format.vgg
            elif (ARGS.format == 'KACHE'):
                self.input_format = datasets.Format.kache

        # For Run in Training Runs
        self.all_training_runs = []
        for trainer in self.trainers:
            self.current_training_dir = os.path.join(self.trainers_dir,trainer)
            ## Prepare Dataset ##
            self.dataset = datasets.DataFormatter(annotations_list = self.annotations_list, input_format =self.input_format,
                                                    output_path = os.path.join(self.current_training_dir, 'data'),
                                                    trainer_prefix = 'COCO_train2014_0000', s3_bucket = self.s3_bucket)
            # Export to Darknet format for training
            self.dataset.export(datasets.Format.darknet)

            # Grab hyperparameters from filename
            tokens = trainer.split('_')
            hyperparams = {'name': tokens[0],
                           'gpus': int(tokens[1].replace('gpu','')),
                           'lr': float('0.'+tokens[2].replace('lr','')),
                           'batch': int(tokens[3].replace('bat','')),
                           'subdivisions': int(tokens[4].replace('sd','')),
                           'epochs': int(tokens[5].replace('ep',''))}

            print('Initiating Trainer:', self.current_training_dir,'\n\n','Hyperameters:', hyperparams)

            # Override data config
            self.current_data_cfg = self.parse_data_config(self.data_cfg)
            os.makedirs(os.path.join(self.current_training_dir, 'data'), exist_ok = True)
            self.current_data_cfg = self.inject_data_config(self.current_data_cfg)
            self.current_data_cfg_path = self.save_data_config(self.current_data_cfg, os.path.join(self.current_training_dir, 'cfg', os.path.split(self.data_cfg)[-1]))

            # Override model config
            self.current_model_cfg = self.parse_model_config(self.model_cfg)
            self.current_model_cfg = self.inject_model_config(self.current_model_cfg, hyperparams)
            os.makedirs(os.path.join(self.current_training_dir, 'cfg'), exist_ok = True)
            self.current_model_cfg_path = self.save_model_config(self.current_model_cfg, os.path.join(self.current_training_dir, 'cfg', os.path.split(self.model_cfg)[-1]))

            # Run Training #
            self.optimize()

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


    def save_model_config(self, model_defs, path):
        """Saves the yolo-v3 layer configuration file"""
        with open(path, 'w') as writer:

            for block in model_defs:
                writer.write('['+ block['type'] +']'+'\n')
                [writer.write(str(k)+'='+str(v)+'\n') for k,v in block.items() if k != 'type']
                writer.write('\n')
        return path


    def save_data_config(self, data_config, path):
        """Saves the yolo-v3 data configuration file"""
        with open(path, 'w') as writer:
            [writer.write(str(k)+'='+str(v)+'\n') for k,v in data_config.items()]
        return path


    def inject_model_config(self, model_config, hyperparams):
        for i, block in enumerate(model_config):
            if block['type'] == 'net':
                block['learning_rate'] = hyperparams['lr']
                block['batch'] = hyperparams['batch']
                block['subdivisions'] = hyperparams['subdivisions']
                block['burn_in'] = len(self.dataset._images.items())//(hyperparams['gpus'] * hyperparams['batch'])
                block['max_batches'] = len(self.dataset._images.items())//(hyperparams['gpus'] * hyperparams['batch']) * hyperparams['epochs']
            elif block['type'] == 'yolo':
                block['classes'] = len(self.dataset.category_names)
                model_config[i-1]['filters'] = (len(self.dataset.category_names)+5)*3
        return model_config


    def inject_data_config(self, data_config):
        data_config['train'] = self.dataset.darknet_manifast
        data_config['classes'] = len(self.dataset.category_names)
        ## TODO: Add Validation Set
        data_config['valid'] = self.dataset.darknet_manifast
        data_config['names'] = self.dataset.names_config
        backup_path = os.path.abspath(os.path.join(self.dataset.output_path, os.pardir, 'backup'))
        os.makedirs(backup_path, exist_ok = True)
        data_config['backup'] = os.path.abspath(os.path.join(self.dataset.output_path, os.pardir, 'backup'))
        num_gpus = int(self.dataset.parse_nvidia_smi()['Attached GPUs'])
        data_config['gpus'] = ','.join(str(i) for i in range(num_gpus))


        return data_config


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


    def get_latest_weights(self):
        # Grab iterations and sort checkpoints
        d = {}
        weights_files = glob.glob(os.path.join(self.current_training_dir, 'backup/', '*.weights'))

        for fpath in weights_files:
            fname = self.dataset.path_leaf(fpath)
            iterations = fname.split('_')[-1].split('.weights')[0]
            if iterations != 'final':
                d[fname] = int(iterations)
            else:
                d[fname] = 1e5
        sorted_weights = OrderedDict(sorted(d.items(), key=lambda kv: kv[1]))

        # Return latest weights
        latest_weights = list(sorted_weights.keys())[-1]
        return os.path.join(self.current_training_dir, 'backup/',  latest_weights)


    def optimize(self):
        try:
            # TODO: Download Darknet 53 weights into backup folder
            self.current_training_results = os.path.join(self.current_training_dir, 'training_results.txt')

            # TODO: Grab weights from backup dir or select convolutional weights
            if self.resume:
                self.current_weights = self.get_latest_weights()
            else:
                self.current_weights =  os.path.join(self.current_training_dir, 'darknet53.conv.74')


            self.num_gpus = self.current_data_cfg['gpus']
            self.darknet_train_cmd = "cd {} && ./darknet detector train {} {} {} -gpus {} | tee -a {}".format(self.current_working_dir,
                                        self.current_data_cfg_path, self.current_model_cfg_path, self.current_weights, self.num_gpus,
                                        self.current_training_results)
            print('Initializing Training with the following parameters:','\n', self.darknet_train_cmd)

            # TODO: Add Evaluation steps
            # TODO: Add early stopping
            proc=Popen(self.darknet_train_cmd, shell=True, stdout=PIPE)
            outfile = self.current_training_results+'.backup'
            with open(outfile,"w+") as f:
                f.write(proc.communicate()[0].decode("utf-8"))

        except KeyboardInterrupt:
            print("\nNow exiting... Results file: " + str(outfile) +
                  "\nCurrent weights: " + str(self.current_weights))
            outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', "--annotations_list", action="store",
                        help="Annotations List", default=ANNOTATIONS_LIST)
    parser.add_argument('-m', "--model_cfg", action="store", default=BASE_MODEL_CONFIG,
                        help="Base model configuration file.")
    parser.add_argument('-d', "--data_cfg", default=BASE_DATA_CONFIG,
                        action="store", help="Base model configuration file.")
    parser.add_argument('-t', "--trainers_dir", default=TRAINERS_DIR,
                        action="store", help="Trainers directory with hyperparamers stored in the paths.")
    parser.add_argument('-b', "--s3_bucket", default=S3_BUCKET,
                        action="store", help="S3 bucket storing the annotations list data.")
    parser.add_argument('-c', "--current_working_dir", default=WORKING_DIR,
                        action="store", help="working directory of darknet files.")
    parser.add_argument('-w', "--weights", default=os.path.join(WORKING_DIR,'backup', 'darknet53.conv.74'),
                        action="store", help="working directory of darknet files.")
    parser.add_argument('-r', "--resume",
                        action="store_true", help="Grabs latests weights from backup and resumes training")
    parser.add_argument('-f', "--format", default='BDD',
                        action="store", help="supported input formats: BDD|COCO|OPENIMGS|SCALABEL|KACHE|VGG")
    args = parser.parse_args()

    # Setup Data Format
    darkernet = Darkernet(args)

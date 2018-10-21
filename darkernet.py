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
VAL_ANNOTATIONS_LIST = os.path.join(BDD100K_DIR, 'labels/bdd100k_labels_images_val.json')

COCO_ANNOTATIONS_LIST = os.path.join('/media/dean/datastore1/datasets/road_coco/darknet/data/coco/annotations/instances_train2014.json')
BASE_DATA_CONFIG = os.path.join(WORKING_DIR, 'cfg', 'bdd100k.data')
BASE_MODEL_CONFIG = os.path.join(WORKING_DIR, 'cfg', 'yolov3-bdd100k.cfg')
S3_BUCKET = 'kache-scalabel/bdd100k/images/100k/train/'


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

        self.train_metrics = OrderedDict()
        self.evaluation_metrics = OrderedDict()
        self.bdd_validation_set = None


        # For Run in Training Runs
        self.all_training_runs = []
        for trainer in self.trainers:
            self.current_training_dir = os.path.join(self.trainers_dir, trainer)
            self.current_weights_dir = os.path.join(self.current_training_dir, 'backup')

            ## Prepare Dataset ##
            self.dataset = datasets.DataFormatter(annotations_list = self.annotations_list, input_format =self.input_format,
                                                    output_path = os.path.join(self.current_training_dir, 'data'),
                                                    trainer_prefix = 'COCO_train2014_0000', s3_bucket = self.s3_bucket)
            # Export to Darknet format for training
            self.dataset.export(datasets.Format.darknet, force = True)

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
            # Run mAP #
            self.evaluate(evaluate_all = True)

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

    def generate_anchors(self, clusters = 9, width = 416, height = 416):
        gen_anchors_cmd = "cd {} && ./darknet detector calc_anchors {} \
        -num_of_clusters {} -width {} -height {}".format(self.current_working_dir, self.current_data_cfg_path, clusters, width, height)
        print('Calculating Anchors from dataset distribution...')
        os.system(gen_anchors_cmd)
        with open(self.anchors_path) as f:
            self.anchors = f.readlines()[0].strip('\'')


    def inject_model_config(self, model_config, hyperparams, validation = False):
        self.anchors_path = os.path.join(self.current_working_dir, 'anchors.txt')
        if not os.path.exists(self.anchors_path):
            self.generate_anchors()
        else:
            with open(self.anchors_path) as f:
                self.anchors = f.readlines()[0].strip('\'')

        for i, block in enumerate(model_config):
            if block['type'] == 'net':
                block['learning_rate'] = hyperparams['lr']
                block['batch'] = hyperparams['batch']
                block['subdivisions'] = hyperparams['subdivisions']
                if validation:
                    block['burn_in'] = len(self.validation_set._images.items())//(hyperparams['gpus'] * hyperparams['batch'])
                    block['max_batches'] = len(self.self.validation_set._images.items())//(hyperparams['gpus'] * hyperparams['batch']) * hyperparams['epochs']
                else:
                    block['burn_in'] = len(self.dataset._images.items())//(hyperparams['gpus'] * hyperparams['batch'])
                    block['max_batches'] = len(self.dataset._images.items())//(hyperparams['gpus'] * hyperparams['batch']) * hyperparams['epochs']
            elif block['type'] == 'yolo':
                if validation:
                    block['classes'] = len(self.validation_set.category_names)
                    block['anchors'] = self.anchors
                    model_config[i-1]['filters'] = (len(self.validation_set.category_names)+5)*3
                else:
                    block['classes'] = len(self.dataset.category_names)
                    block['anchors'] = self.anchors
                    model_config[i-1]['filters'] = (len(self.dataset.category_names)+5)*3
        return model_config


    def inject_data_config(self, data_config, validation = False):
        if validation:
            data_config['train'] = self.validation_set.darknet_manifast
            data_config['classes'] = len(self.validation_set.category_names)
            data_config['valid'] = self.validation_set.darknet_manifast
            data_config['names'] = self.validation_set.names_config
            backup_path = os.path.abspath(os.path.join(self.validation_set.output_path, os.pardir, 'backup'))
            os.makedirs(backup_path, exist_ok = True)
            data_config['backup'] = os.path.abspath(os.path.join(self.validation_set.output_path, os.pardir, 'backup'))
            num_gpus = int(self.validation_set.parse_nvidia_smi()['Attached GPUs'])
            data_config['gpus'] = ','.join(str(i) for i in range(num_gpus))
        else:
            data_config['train'] = self.dataset.darknet_manifast
            data_config['classes'] = len(self.dataset.category_names)
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

    def generate_validation_set(self):
        val = datasets.DataFormatter(annotations_list = VAL_ANNOTATIONS_LIST,
                                    input_format = datasets.Format.bdd,
                                    output_path = os.path.join(self.current_training_dir, 'data', 'val'),
                                    trainer_prefix = 'COCO_val2014_0000',
                                    s3_bucket = 'kache-scalabel/bdd100k/images/100k/val/')
        val.export(datasets.Format.darknet, force = True)
        return val

    def get_latest_weights(self):
        """Grab iterations and sort checkpoints"""
        d = {}
        weights_files = glob.glob(os.path.join(self.current_training_dir, 'backup/', '*.weights'))
        init_weight = glob.glob(os.path.join(self.current_training_dir, 'backup/', '*.74'))
        weights_files.extend(init_weight)

        for fpath in weights_files:
            fname = self.dataset.path_leaf(fpath)
            iterations = fname.split('_')[-1].split('.weights')[0]
            if iterations == 'darknet53.conv.74':
                d[fname] = 0
            elif iterations != 'final':
                d[fname] = int(iterations)
            else:
                d[fname] = 1e5
        self.sorted_weights = OrderedDict(sorted(d.items(), key=lambda kv: kv[1]))

        # Return latest weights
        latest_weights = list(self.sorted_weights.keys())[-1]
        return os.path.join(self.current_training_dir, 'backup/',  latest_weights)


    def evaluate(self, evaluate_all = False):
        """ Generates a validation set in BDD format and then Runs mAP against backup_dir"""
        self.current_train_metrics = OrderedDict()
        if not self.bdd_validation_set:
            self.bdd_validation_set = self.generate_validation_set()
        try:
            bdd_weights = [self.get_latest_weights()]
            if evaluate_all:
                bdd_weights = list(self.sorted_weights.keys())

            for weights in bdd_weights:
                self.current_train_metrics['iterations']    = self.sorted_weights[self.dataset.path_leaf(weights)]
                self.current_train_metrics['map_results_file'] = weights+'.txt'
                map_results_file = self.current_train_metrics['map_results_file']
                self.current_map_results = self.current_train_metrics['map_results_file'] +'.backup'
                if not os.path.exists(self.current_map_results):
                    self.darknet_map_cmd = "cd {} && ./darknet detector map {} {} {} | tee -a {}".format(self.current_working_dir,
                                                self.current_data_cfg_path, self.current_model_cfg_path, weights, map_results_file)
                    print('Initializing Evaluation with the following parameters:','\n', self.darknet_map_cmd)
                    proc=Popen(self.darknet_map_cmd, shell=True, stdout=PIPE)
                    outfile = self.current_train_metrics['map_results_file']+'.backup'
                    with open(outfile,"w+") as f:
                        f.write(proc.communicate()[0].decode("utf-8"))

                self.current_train_metrics['evaluation_results'] = self.collect_evaluation_metrics(weights)

            self.train_metrics[self.dataset.path_leaf(self.current_weights)] = self.current_train_metrics
        except KeyboardInterrupt:
            print("\nNow exiting evaluation... Results file: " + str(map_results_file) +
                  "\nCurrent weights: " + str(self.get_latest_weights()))
            f.close()

    def collect_evaluation_metrics(self, weights):
        """Parses the darknet mAP output and collects relevant metrics"""
        #Get mAP results from file
        class_stats = []
        map_stats = {}
        with open(self.current_map_results) as openfile:
            for line in openfile:
                for part in line.split():
                    if 'class_id' in part:
                        class_map = {}
                        if 'class_id =' in line:
                            id_tokens = line.split('class_id =')[1].split(',')
                            if id_tokens[0] in [str(i) for i in range(10)]:
                                class_map['class_id'] = int(id_tokens[0].strip())

                        if 'name =' in line:
                            name_tokens = line.split('name =')[1].split(',')
                            class_map['class_name'] = name_tokens[0].strip()

                        if 'ap =' in line:
                            ap_tokens = line.split('ap =')[1].split(',')
                            class_map['class_ap'] = ap_tokens[0].split('%')[0].strip()

                        class_stats.append(class_map)
                    elif '(mAP)' in part:
                        tokens = line.split('(mAP) =')[1]
                        map_stats['mean_avg_precision'] = tokens.split()[2].strip(',')
                    elif 'Detection' in part:
                        tokens = line.split(':')[1]
                        total_detection_time = str(''.join(tokens)).strip()
                        map_stats['total_detection_time'] = total_detection_time
                    elif 'average IoU' and  'average IoU =' in line:
                        tokens = line.split('average IoU =')[1].split(',')
                        map_stats['avg_IoU'] = tokens[0].split('%')[0].strip()

        map_stats['class_stats'] = class_stats
        print(map_stats,'\n\n')
        self.evaluation_metrics[weights] = map_stats


        # Cache Data
        pickle_file = os.path.join(self.current_training_dir,'evaluation_metrics.pickle')
        pickle_dict = {'evaluation_metrics': self.evaluation_metrics}
        with open(pickle_file,"wb") as pickle_out:
            pickle.dump(pickle_dict, pickle_out)


    def optimize(self):
        try:
            self.current_train_metrics = OrderedDict()
            if not os.path.exists(os.path.join(self.current_training_dir, 'backup', 'darknet53.conv.74')):
                darknet53_path = 'https://pjreddie.com/media/files/darknet53.conv.74'
                res = os.system("curl -o {} {}".format(os.path.join(self.current_training_dir, 'backup', 'darknet53.conv.74'), darknet53_path))
            self.current_training_results = os.path.join(self.current_training_dir, 'training_results.txt')

            if self.resume:
                self.current_weights = self.get_latest_weights()
            else:
                self.current_weights =  os.path.join(self.current_training_dir, 'backup', 'darknet53.conv.74')

            self.num_gpus = self.current_data_cfg['gpus']
            self.darknet_train_cmd = "cd {} && ./darknet detector train {} {} {} -gpus {} | tee -a {}".format(self.current_working_dir,
                                        self.current_data_cfg_path, self.current_model_cfg_path, self.current_weights, self.num_gpus,
                                        self.current_training_results)
            print('Initializing Training with the following parameters:','\n', self.darknet_train_cmd)
            proc=Popen(self.darknet_train_cmd, shell=True, stdout=PIPE)
            outfile = self.current_training_results+'.backup'
            with open(outfile,"w+") as f:
                f.write(proc.communicate()[0].decode("utf-8"))

            # TODO: Add Evaluation procedure
            # TODO: Add early stopping
            # Run Validation #
            #self.evaluate()

        except KeyboardInterrupt:
            print("\nNow exiting... Results file: " + str(outfile) +
                  "\nCurrent weights: " + str(self.get_latest_weights()))
            f.close()


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

"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa

Reads in a tsv file with pre-trained bottom up attention features 
of the adaptive number of boxes and stores it in HDF5 format.  
Also store {image_id: feature_idx} as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_boxes x 2048
  'image_bb': num_boxes x 4
  'spatial_features': num_boxes x 6
  'pos_boxes': num_images x 2 }
"""
from __future__ import print_function
from collections import defaultdict
from email.mime import image

import os
from os import listdir
from os import path
from os.path import isfile, join
import argparse
from posixpath import split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import csv
import h5py
import _pickle as cPickle
import numpy as np
import utils
import csv
import pandas as pd 
import base64


def create_mapping(labels_file):
    '''
    This function creates the mapping function from the old classes to the new ones.
    :param labels_file: new classes.
    :return: mapping function, index to labels name for new classes, index to labels name for old classes
    '''
    # loading cleaned classes
    print("Loading cleaned Visual Genome classes: {} .".format(labels_file))
    with open(labels_file, 'r') as file:
        cleaned_labels = file.readlines()
    # remove new line symbol and leading/trailing spaces.
    cleaned_labels = [i.strip('\n').strip() for i in cleaned_labels]
    # make dictionary
    cleaned_labels = {id+1: label for id, label in enumerate(cleaned_labels)}     # [1, 1600]
    # get previously labels from the same file and make the mapping function
    map_fn = dict()
    old_labels = dict()
    for new_label_id, new_label_str in cleaned_labels.items():
        new_label_id = int(new_label_id)
        for piece in new_label_str.split(','):
            tmp = piece.split(':')
            assert len(tmp) == 2
            old_label_id = int(tmp[0])
            old_label_str = tmp[1]
            # we need to avoid overriding of same ids like: 17:stop sign,17:stopsign
            if old_label_id not in old_labels.keys():
                old_labels[old_label_id] = old_label_str
                map_fn[old_label_id] = new_label_id
            else:
                print('Warning: label already present for {}:{}. Class {} ignored. '.format(old_label_id,
                                                                                            old_labels[old_label_id],
                                                                                            old_label_str))
    assert len(old_labels) == 1600
    assert len(old_labels) == len(map_fn)
    # print(old_labels[1590], map_fn[1590], cleaned_labels[map_fn[1590]])
    return map_fn, cleaned_labels, old_labels     # all in [1, 1600]

def load_data(img_folder, labels_path, classes_type, model_type):
    print("Considering just classes type: ", classes_type)
    print("Model type: ", model_type)
    max_number_of_classes = 877 if model_type =='cleaned' else 1599
    # get new classes labels
    if classes_type != 'all':
        map_fn, _, _ = create_mapping(labels_path)
        map_fn_reverse = defaultdict(list)
        for k, v in map_fn.items():
            map_fn_reverse[v].append(k)
        if model_type == 'noisy':
            untouched_cls_idx = {v[0]: k for k, v in map_fn_reverse.items() if len(v) == 1}
        elif model_type == 'cleaned':
            untouched_cls_idx = {k: v[0]  for k, v in map_fn_reverse.items() if len(v) == 1}
        else:
            print('Error in model type: ', model_type)
            exit(1)
        untouched_cls_idx = [k-1 for k, v in untouched_cls_idx.items()]

    # get all extracted file in the folder
    onlyfiles = [join(img_folder, f) for f in listdir(img_folder) if isfile(join(img_folder, f))]
    print('Number of files: ', len(onlyfiles))
    onlyfiles = [f for f in onlyfiles if f[-4:] == '.npz']
    print('Number of .npz files: ', len(onlyfiles))

    # load all data ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
    all_data = defaultdict(list)
    count_zeros = 0
    for img_file in onlyfiles:
        img_id = img_file.split('/')[-1][:-4]
        with np.load(img_file, allow_pickle=True) as f:
            data_info = f['info'].item() # check https://stackoverflow.com/questions/40219946/python-save-dictionaries-through-numpy-save
            # info = {
            #     "objects": classes.cpu().numpy(),
            #     "cls_prob": cls_probs.cpu().numpy(),
            #     'attrs_id': attr_probs,
            #     'attrs_scores': attr_scores,
            # }
            data_num_bbox = f['num_bbox']
            data_boxes = f['bbox']
            data_features = f['x']
            assert data_num_bbox == len(data_info['objects']) == len(data_info['cls_prob']) == len(data_boxes) == len(data_features)
            assert img_id not in all_data['image_id']

            # bounding boxes filtering according to its label 
            filtered_boxes = []
            filtered_features = []
            for box_idx in range(data_num_bbox):
                box_label_idx = data_info['objects'][box_idx]  # in [0, 877 or 1599]
                assert len(data_boxes[box_idx]) == 4
                assert 0 <= box_label_idx <= max_number_of_classes
                # NOTE: BE SURE EVERYTHING IS np.float32 WHEN DEALING WITH base64.b64encode() function
                if classes_type == 'untouched':
                    if box_label_idx in untouched_cls_idx:
                        filtered_boxes.append(data_boxes[box_idx])
                        filtered_features.append(data_features[box_idx])
                elif classes_type == 'new':
                    if box_label_idx not in untouched_cls_idx:
                        filtered_boxes.append(data_boxes[box_idx])
                        filtered_features.append(data_features[box_idx])
                elif classes_type == 'all':
                    filtered_features.append(data_features[box_idx])
                    filtered_boxes.append(data_boxes[box_idx])
                else:
                    print('Error.')
                    exit(1)
            
            if len(filtered_boxes) == 0:
                filtered_boxes.append(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
                filtered_features.append(data_features[0])
                count_zeros += 1

            all_data['image_id'].append(img_id)
            all_data['image_w'].append(f['image_w'])
            all_data['image_h'].append(f['image_h'])
            all_data['num_boxes'].append(len(filtered_boxes))
            # need to be encoded. See adaptive_detection_features_converter.py
            all_data['boxes'].append(base64.b64encode(np.array(filtered_boxes)))  # need to be encoded. See adaptive_detection_features_converter.py
            all_data['features'].append(base64.b64encode(np.array(filtered_features)))
            # all_data['image_h_inner'].append(f['image_h_inner'])
            # all_data['image_w_inner'].append(f['image_w_inner'])
            # all_data['info'].append(f['info'])
        
    print("Number of images with zero boxes: ", count_zeros)
    return pd.DataFrame(all_data)


def save_tsv(subset_data, output_folder, split_name):
    file_name = '{}_flickr30k_resnet101_faster_rcnn_genome.tsv'.format(split_name)
    file_path = os.path.join(output_folder, file_name)
    # with open(file_path, 'w') as tsvfile:
    #       writer = csv.writer(tsvfile, delimiter='\t', newline='\n')
    #       pd.DataFrame(np_array)
    subset_data.to_csv(file_path, sep="\t",header=False, index=False)
    print('Data saved: ', file_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_features', type=str, default='./data/flickr30k/extracted_features/', help='Folder of extracted features')
    parser.add_argument('--output_folder', type=str, default='./data/flickr30k/', help='Folder where to save the .tsv file.')
    parser.add_argument('--labels', dest='labels',
                    help='File containing the new cleaned labels. It is needed for extracting the old and new classes indexes.',
                    default="./data/objects_vocab.txt",
                    type=str)
    parser.add_argument('--classes', dest='classes',
                help='Classes to consider.',
                default='all',
                choices=['all', 'untouched', 'new'],
                type=str)
    parser.add_argument('--model', dest='model',
            help='Model trained on new classes (878 labels) or model post-processed (1600 to 878 labels).',
            default='noisy',
            choices=['noisy', 'cleaned'],
            type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # check if the folder exists
    if os.path.exists(args.extracted_features):
        print('Loading all data.')
        all_data = load_data(args.extracted_features, args.labels, args.classes, args.model)
        
        print("Mean number of boxes: ", np.mean(all_data['num_boxes']))
        print("Max number of boxes: ", max(all_data['num_boxes']))
        print("Min number of boxes: ", min(all_data['num_boxes']))

        print("Saving data.")
        splits = ['./data/flickr30k/flickr30k_entities/train.txt',
                    './data/flickr30k/flickr30k_entities/val.txt',
                    './data/flickr30k/flickr30k_entities/test.txt']
        for split_path in splits:
            split_name = split_path.split('/')[-1][:-4]
            with open(split_path, 'r') as f:
                split_idx = f.read().splitlines() 
            subset_data = all_data[all_data['image_id'].isin(split_idx)]
            print('Processing split {} with {}/{} idx found. '.format(split_name, len(subset_data), len(split_idx)))
            save_tsv(subset_data, args.output_folder, split_name)
    else:
        print("Folder not valid: ", args.extracted_features)
        exit(1)
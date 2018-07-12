"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
import argparse
import math
from matching import Matching
import csv
from rectangle import Rectangle
import time

"""
  Scoring code to calculate per-class precision and mean average precision.

  Args:
      predictions: a folder path of prediction files.  
        Prediction files should have filename format 'XYZ.tif.txt',
        where 'XYZ.tif' is the xView TIFF file being predicted on.  
        Prediction files should be in space-delimited csv format, with each
        line like (xmin ymin xmax ymax class_prediction score_prediction).
        ie ("predictions/")

      groundtruth: a filepath to ground truth labels (GeoJSON format)
        ie ("ground_truth.geojson")

      output (-o): a folder path where output metrics are saved
        ie ("scores/")

  Outputs:
    Writes two files to the 'output' parameter folder: 'score.txt' and 'metrics.txt'
    'score.txt' contains a single floating point value output: mAP
    'metrics.txt' contains the remaining metrics in per-line format (metric/class_num: score_float)

"""

def get_labels(fname):
  """
    Processes a WorldView3 GEOJSON file

    Args:
        fname: filepath to the GeoJson file.

    Outputs:
      Bounding box coordinate array, Chip-name array, and Classes array

  """
  with open(fname) as f:
      data = json.load(f)

  coords = np.zeros((len(data['features']),4))
  chips = np.zeros((len(data['features'])),dtype="object")
  classes = np.zeros((len(data['features'])))

  for i in tqdm(range(len(data['features']))):
      if data['features'][i]['properties']['bounds_imcoords'] != []:
          b_id = data['features'][i]['properties']['image_id']
          val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
          chips[i] = b_id
          classes[i] = data['features'][i]['properties']['type_id']
          if val.shape[0] != 4:
              raise ValueError('A bounding box should have 4 entries!')
          else:
              coords[i] = val
      else:
          chips[i] = 'None'

  return coords, chips, classes

def convert_to_rectangle_list(coordinates):
  """
    Converts a list of coordinates to a list of rectangles

    Args:
        coordinates: a flattened list of bounding box coordinates in format
          (xmin,ymin,xmax,ymax)

    Outputs:
      A list of rectangles

  """
  rectangle_list = []
  number_of_rects = int(len(coordinates) / 4)
  for i in range(number_of_rects):
    rectangle_list.append(Rectangle(
        coordinates[4 * i], coordinates[4 * i + 1], coordinates[4 * i + 2],
        coordinates[4 * i + 3]))
  return rectangle_list

def ap_from_pr(p,r):
  """
    Calculates AP from precision and recall values as specified in
    the PASCAL VOC devkit.

    Args:
        p: an array of precision values
        r: an array of recall values

    Outputs:
      An average precision value

  """
  r = np.concatenate([[0], r, [1]])
  p = np.concatenate([[0], p, [0]])
  for i in range(p.shape[0] - 2, 0, -1):
    if p[i] > p[i-1]:
      p[i-1] = p[i]

  i = np.where(r[1:] != r[:len(r)-1])[0] + 1
  ap = np.sum(
      (r[i] - r[i - 1]) * p[i])

  return ap

def score(path_predictions, path_groundtruth, path_output, iou_threshold = .5):
  """
    Compute metrics on a number of prediction files, given a folder of prediction files
    and a ground truth.  Primary metric is mean average precision (mAP).

    Args:
        path_predictions: a folder path of prediction files.  
          Prediction files should have filename format 'XYZ.tif.txt',
          where 'XYZ.tif' is the xView TIFF file being predicted on.  
          Prediction files should be in space-delimited csv format, with each
          line like (xmin ymin xmax ymax class_prediction score_prediction)

        path_groundtruth: a file path to a single ground truth geojson

        path_output: a folder path for output scoring files

        iou_threshold: a float between 0 and 1 indicating the percentage
          iou required to count a prediction as a true positive

    Outputs:
      Writes two files to the 'path_output' parameter folder: 'score.txt' and 'metrics.txt'
      'score.txt' contains a single floating point value output: mAP
      'metrics.txt' contains the remaining metrics in per-line format (metric/class_num: score_float)

    Raises:
      ValueError: if there are files in the prediction folder that are not in the ground truth geojson.
        EG a prediction file is titled '15.tif.txt', but the file '15.tif' is not in the ground truth.

  """
  assert (iou_threshold < 1 and iou_threshold > 0)

  ttime = time.time()
  boxes_dict = {}
  pchips = []
  stclasses = []
  num_preds = 0

  for file in tqdm(os.listdir(path_predictions)):
      fname = file.split(".txt")[0]
      pchips.append(fname)

      with open(path_predictions + file,'r') as f:
        arr = np.array(list(csv.reader(f,delimiter=" ")))
        if arr.shape[0] == 0:
            #If the file is empty, we fill it in with an array of zeros
            boxes_dict[fname] = np.array([[0,0,0,0,0,0]])
            num_preds += 1
        else:
            arr = arr[:,:6].astype(np.float64)
            threshold = 0
            arr = arr[arr[:,5] > threshold]
            stclasses += list(arr[:,4])
            num_preds += arr.shape[0]

            if np.any(arr[:,:4] < 0):
              raise ValueError('Bounding boxes cannot be negative.')

            if np.any(arr[:,5] < 0) or np.any(arr[:,5] > 1):
              raise ValueError('Confidence scores should be between 0 and 1.')

            boxes_dict[fname] = arr[:,:6]
 
  pchips = sorted(pchips)
  stclasses = np.unique(stclasses).astype(np.int64)

  gt_coords, gt_chips, gt_classes = get_labels(path_groundtruth)

  gt_unique = np.unique(gt_classes.astype(np.int64))
  max_gt_cls = 100


  if set(pchips).issubset(set(gt_unique)):
      raise ValueError('The prediction files {%s} are not in the ground truth.' % str(set(pchips) - (set(gt_unique))))

  print("Number of Predictions: %d" % num_preds)
  print("Number of GT: %d" % np.sum(gt_classes.shape) )


  per_file_class_data = {}
  for i in gt_unique:
    per_file_class_data[i] = [[],[]]

  num_gt_per_cls =  np.zeros((max_gt_cls))

  for file_ind in range(len(pchips)):
      print(pchips[file_ind])
      det_box = boxes_dict[pchips[file_ind]][:,:4]
      det_scores = boxes_dict[pchips[file_ind]][:,5]
      det_cls = boxes_dict[pchips[file_ind]][:,4]

      gt_box = gt_coords[(gt_chips==pchips[file_ind]).flatten()]
      gt_cls = gt_classes[(gt_chips==pchips[file_ind])]
      
      for i in gt_unique:
        s = det_scores[det_cls == i]
        ssort = np.argsort(s)[::-1]
        per_file_class_data[i][0] += s[ssort].tolist()

        gt_box_i_cls = gt_box[gt_cls == i].flatten().tolist()
        det_box_i_cls = det_box[det_cls == i]
        det_box_i_cls = det_box_i_cls[ssort].flatten().tolist()

        gt_rects = convert_to_rectangle_list(gt_box_i_cls)
        rects = convert_to_rectangle_list(det_box_i_cls)

        matching = Matching(gt_rects, rects)
        rects_matched, gt_matched = matching.greedy_match(iou_threshold)

        #we aggregate confidence scores, rectangles, and num_gt across classes 
        #per_file_class_data[i][0] += det_scores[det_cls == i].tolist()
        per_file_class_data[i][1] += rects_matched
        num_gt_per_cls[i] += len(gt_matched)

  average_precision_per_class = np.ones(max_gt_cls) * float('nan')
  per_class_p = np.ones(max_gt_cls) * float('nan')
  per_class_r = np.ones(max_gt_cls) * float('nan')

  xview_categories_all = [
    11, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 71, 72, 73, 74, 76, 77, 79, 83, 84, 86, 89, 91, 93, 94,
    ]

  # there are 3 extra classes in gt_unique which should not be scored in xView datasets: 75, 82, 87
  # Remove those classes if they are present
  ignored_classes = [75, 82, 87]
  gt_unique_xv = np.array([i for i in gt_unique if int(i) not in ignored_classes], dtype = np.int64)

  for i in gt_unique_xv:
    scores = np.array(per_file_class_data[i][0])
    rects_matched = np.array(per_file_class_data[i][1])

    if num_gt_per_cls[i] != 0:
      sorted_indices = np.argsort(scores)[::-1]
      tp_sum = np.cumsum(rects_matched[sorted_indices])
      fp_sum = np.cumsum(np.logical_not(rects_matched[sorted_indices]))
      precision = tp_sum / (tp_sum + fp_sum + np.spacing(1))
      recall = tp_sum / num_gt_per_cls[i]
      per_class_p[i] = np.sum(rects_matched) / len(rects_matched)
      per_class_r[i] = np.sum(rects_matched) / num_gt_per_cls[i]
      ap = ap_from_pr(precision,recall)
    else:
      # ap = float('nan')
      ap = 0.0 # So that predicting 'NaN' for a whole category doesn't artificially boost scores
    average_precision_per_class[i] = ap

  #metric splits
  metric_keys = ['map','map/small','map/medium','map/large',
  'map/common','map/rare']

  splits = {
  'map/small': [17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 32, 41, 60,
                   62, 63, 64, 65, 66, 91],
  'map/medium': [11, 12, 15, 25, 29, 33, 34, 35, 36, 37, 38, 42, 44,
                  47, 50, 53, 56, 59, 61, 71, 72, 73, 76, 84, 86, 93, 94],
  'map/large': [13, 40, 45, 49, 51, 52, 54, 55, 57, 74, 77, 79, 83, 89],

  'map/common': [13,17,18,19,20,21,23,24,25,26,27,28,34,35,41,
                  47,60,63,64,71,72,73,76,77,79,83,86,89,91],
  'map/rare': [11,12,15,29,32,33,36,37,38,40,42,44,45,49,50,
                  51,52,53,54,55,56,57,59,61,62,65,66,74,84,93,94]
  }

  vals = {}
  vals['map'] = np.nanmean(average_precision_per_class)
  vals['map_score'] = np.nanmean(per_class_p)
  vals['mar_score'] = np.nanmean(per_class_r)

  for i in splits.keys():
    vals[i] = np.nanmean(average_precision_per_class[splits[i]])

  for i in gt_unique:
    vals[int(i)] = average_precision_per_class[int(i)]

  vals['f1'] =  2 /  ( (1 / (np.spacing(1) + vals['map_score']) ) 
    + ( 1 / ( np.spacing(1) + vals['mar_score'])) )

  print("mAP: %f | mAP score: %f | mAR: %f | F1: %f" % 
    (vals['map'],vals['map_score'],vals['mar_score'],vals['f1']))

  with open(path_output + '/score.txt','w') as f:
      f.write(str("%.8f" % vals['map']))

  with open(path_output + '/metrics.txt','w') as f:
      for key in vals.keys():
          f.write("%s %f\n" % (str(key),vals[key]) )

  print("Final time: %s" % str(time.time() - ttime))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="Path to properly formatted predictions file")
    parser.add_argument("groundtruth", help="Path to groundtruth GeoJSON file")
    parser.add_argument('-o',"--output",default=".",
        help="Output path for calculated scores")
    args = parser.parse_args()

    score(args.predictions, args.groundtruth, args.output)
    

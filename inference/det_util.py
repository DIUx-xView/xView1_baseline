# Original work Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2018 Defense Innovation Unit Experimental.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
from tqdm import tqdm

def generate_detections(checkpoint,images):
    
    print("Creating Graph...")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    boxes = []
    scores = []
    classes = []
    k = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_np in tqdm(images):
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                box = detection_graph.get_tensor_by_name('detection_boxes:0')
                score = detection_graph.get_tensor_by_name('detection_scores:0')
                clss = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (box, score, clss, num_detections) = sess.run(
                        [box, score, clss, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                boxes.append(box)
                scores.append(score)
                classes.append(clss)
                
    boxes =   np.squeeze(np.array(boxes))
    scores = np.squeeze(np.array(scores))
    classes = np.squeeze(np.array(classes))

    return boxes,scores,classes
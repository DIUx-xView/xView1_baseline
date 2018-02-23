#!/bin/bash

TESTLIST="X.tif Y.tif"

for i in $TESTLIST; do
    echo $i
    python create_detections.py -c ../models/frozen_model.pb -o 'preds_output/'$i'.txt' 'xView/rgb/'$i
done

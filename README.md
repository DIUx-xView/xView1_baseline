## xView Baselines and Scoring

### Code

This repository contains the pre-trained xView baseline models (see our [blog post](https://medium.com/@dariusl/object-detection-baselines-in-overhead-imagery-with-diux-xview-c39b1852f24f)) as well as code for inference and scoring.  Class ID to name mappings are in the 'xview_class_labels.txt' file

Inference and scoring code are under the 'inference/' and 'scoring/' folders, respectively.  Inside 'inference/' we provide a python script 'create_detections.py' for exporting detections from a single xView TIF image, given a frozen model.  There is also a script 'create_detections.sh' for exporting detections from multiple xView TIF images.  Exported detections are in format


|X min|Y min|X max|Y max|Class ID|Confidence|
|---|---|---|---|---|---|


which is the proper format for submitting to the xView challenge portal.  You can use the given pre-trained baseline models for the '-c' checkpoint parameter.  

The 'scoring/' folder contains code for evaluating a set of predictions (exported from 'create_detections.py' or 'create_detections.sh') given a ground truth label geojson.  The script 'score.py' calculates scores: total mean average precision (mAP), per-class mAP, mAP for small/med/large classes, mAP for rare/common classes, F1 score, mean precision, and mean recall. We use the PASCAL VOC method for computing mean average precision and calculate mean precision and mean recall using the formulas:

Precision = (True Positives) / (True Positives + False Positives)

Recall = (True Positives) / (True Positives + False Negatives)

averaged over all classes and files. Class splits are shown at the bottom of this README.

Pre-trained baseline models can be found in a zip file under the 'releases' tab.  There are three models: vanilla, multires, and multires_aug which are described [here](https://medium.com/@dariusl/object-detection-baselines-in-overhead-imagery-with-diux-xview-c39b1852f24f).  Inside the zip file are three folders and three pb files.  The pb files are frozen models: they can be plugged into the detection scripts right away.  Inside each respective folder are the tensorflow checkpoint files that can be used for fine-tuning.  


### Class Splits

Small:
['Passenger Vehicle', 'Small car', 'Bus', 'Pickup Truck', 'Utility Truck', 'Truck', 'Cargo Truck', 'Truck Tractor', 'Trailer', 'Truck Tractor w/ Flatbed Trailer', 'Crane Truck', 'Motorboat', 'Dump truck', 'Tractor', 'Front loader/Bulldozer', 'Excavator', 'Cement mixer', 'Ground grader', 'Shipping container']

Medium:
['Fixed-wing aircraft', 'Small aircraft', 'Helicopter', 'Truck Tractor w/ Box Trailer', 'Truck Tractor w/ Liquid Tank', 'Railway vehicle', 'Passenger car', 'Cargo/container car', 'Flat car', 'Tank car', 'Locomotive', 'Sailboat', 'Tugboat', 'Fishing vessel', 'Yacht', 'Engineering vehicle', 'Reach stacker', 'Mobile crane', 'Haul truck', 'Hut/Tent', 'Shed', 'Building', 'Damaged/demolished building', 'Helipad', 'Storage Tank', 'Pylon', 'Tower']


Large:
['Passenger/cargo plane', 'Maritime vessel', 'Barge', 'Ferry', 'Container ship', 'Oil Tanker', 'Tower crane', 'Container crane', 'Straddle carrier', 'Aircraft Hangar', 'Facility', 'Construction site', 'Vehicle Lot', 'Shipping container lot']

---

Common:
['Passenger/cargo plane', 'Passenger Vehicle', 'Small car', 'Bus', 'Pickup Truck', 'Utility Truck', 'Truck', 'Cargo Truck', 'Truck Tractor w/ Box Trailer', 'Truck Tractor', 'Trailer', 'Truck Tractor w/ Flatbed Trailer', 'Passenger car', 'Cargo/container car', 'Motorboat', 'Fishing vessel', 'Dump truck', 'Front loader/Bulldozer', 'Excavator', 'Hut/Tent', 'Shed', 'Building', 'Damaged/demolished building', 'Facility', 'Construction site', 'Vehicle Lot', 'Storage Tank', 'Shipping container lot', 'Shipping container']

Rare:
['Fixed-wing aircraft', 'Small aircraft', 'Helicopter', 'Truck Tractor w/ Liquid Tank', 'Crane Truck', 'Railway vehicle', 'Flat car', 'Tank car', 'Locomotive', 'Maritime vessel', 'Sailboat', 'Tugboat', 'Barge', 'Ferry', 'Yacht', 'Container ship', 'Oil Tanker', 'Engineering vehicle', 'Tower crane', 'Container crane', 'Reach stacker', 'Straddle carrier', 'Mobile crane', 'Haul truck', 'Tractor', 'Cement mixer', 'Ground grader', 'Aircraft Hangar', 'Helipad', 'Pylon', 'Tower']

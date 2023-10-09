#!/bin/bash

python testing.py -n 12_3_3_swav_best -d 2 -fast 0 -p /path/to/model
python testing.py -n 12_3_3_swav_best -d 2 -fast 1
python testing.py -n 12_3_3_swav_skip_cov_best -d 2 -fast 1
python testing.py -n 12_3_3_resnet50_best -d 2 -fast 1
python testing.py -n 12_3_3_resnet50_skip_cov_best -d 2 -fast 1
python testing.py -n 12_3_3_scratch_best -d 2 -fast 1


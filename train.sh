#!/bin/bash

python ModelTrainer.py -n 12_3_3_swav              -device 0 -b resnet50_swav -epoch_pre 15 -epoch_fine 40 -batch_pre 36 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6 -batch_trfm 1 -aug_light 1
python ModelTrainer.py -n 12_3_3_swav_skip_cov     -device 0 -b resnet50_swav -epoch_pre 15 -epoch_fine 40 -batch_pre 36 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6 -batch_trfm 1 -aug_light 1 -skip_cov 1
python ModelTrainer.py -n 12_3_3_resnet50          -device 0 -b resnet50      -epoch_pre 15 -epoch_fine 40 -batch_pre 36 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6 -batch_trfm 1 -aug_light 1
python ModelTrainer.py -n 12_3_3_resnet50_skip_cov -device 0 -b resnet50      -epoch_pre 15 -epoch_fine 40 -batch_pre 36 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6 -batch_trfm 1 -aug_light 1 -skip_cov 1
python ModelTrainer.py -n 12_3_3_scratch           -device 0 -b scratch       -epoch_pre 15 -epoch_fine 40 -batch_pre 36 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6 -batch_trfm 1 -aug_light 1

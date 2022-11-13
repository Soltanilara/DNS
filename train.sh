#!/bin/bash

#python ModelTrainer.py -n base -r 0 -j 0 -d 0
#
#python ModelTrainer.py -n r -r 1 -j 0 -d 0
#python ModelTrainer.py -n j -r 0 -j 1 -d 0
#python ModelTrainer.py -n d -r 0 -j 0 -d 1
#
#python ModelTrainer.py -n rj -r 1 -j 1 -d 0
#python ModelTrainer.py -n rd -r 1 -j 0 -d 1
#python ModelTrainer.py -n jd -r 1 -j 0 -d 1
#
#python ModelTrainer.py -n rjd -r 1 -j 1 -d 1

#May 16
#python ModelTrainer.py -n size_1 -device 3 -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 128 -batch_fine 8 -size_sup 1 -size_qry 1 -num_qry 8
#python ModelTrainer.py -n size_3 -device 3 -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 42 -batch_fine 2 -size_sup 3 -size_qry 3 -num_qry 8

#May 21
#python ModelTrainer.py -n efficientnet-b0 -device 3 -b efficientnet-b0 -aug_j 1 -epoch_pre 5 -epoch_fine 60 -batch_pre 112 -batch_fine 8 -size_sup 1 -size_qry 1 -num_qry 6
#python ModelTrainer.py -n efficientnet-b1 -device 3 -b efficientnet-b1 -aug_j 1 -epoch_pre 5 -epoch_fine 60 -batch_pre 96 -batch_fine 6 -size_sup 1 -size_qry 1 -num_qry 6
#python ModelTrainer.py -n efficientnet-b2 -device 3 -b efficientnet-b2 -aug_j 1 -epoch_pre 5 -epoch_fine 60 -batch_pre 96 -batch_fine 6 -size_sup 1 -size_qry 1 -num_qry 6
#python ModelTrainer.py -n efficientnet-b3 -device 3 -b efficientnet-b3 -aug_j 1 -epoch_pre 5 -epoch_fine 60 -batch_pre 80 -batch_fine 6 -size_sup 1 -size_qry 1 -num_qry 6
#python ModelTrainer.py -n efficientnet-b4 -device 3 -b efficientnet-b4 -aug_j 1 -epoch_pre 5 -epoch_fine 60 -batch_pre 80 -batch_fine 6 -size_sup 1 -size_qry 1 -num_qry 6

#May 22
#python ModelTrainer.py -n efficientnet-b0_size_10 -device 0 -b efficientnet-b0 -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 24 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6  # VRAM 48313MB
#python ModelTrainer.py -n efficientnet-b1_size_10 -device 0 -b efficientnet-b1 -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 24 -batch_fine 2 -size_sup 10 -size_qry 10 -num_qry 6  # VRAM 40641MB
#python ModelTrainer.py -n efficientnet-b2_size_10 -device 0 -b efficientnet-b2 -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 24 -batch_fine 2 -size_sup 10 -size_qry 10 -num_qry 6  # VRAM 40645MB

#May 25
#python ModelTrainer.py -n resnet-50               -device 0 -b resnet50        -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 1 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6  # VRAM 5288-2093MB (pre)
#python ModelTrainer.py -n efficientnet-b0_size_10 -device 0 -b efficientnet-b0 -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 1 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6  # VRAM 6114-2093MB (pre)
#python ModelTrainer.py -n efficientnet-b1_size_10 -device 0 -b efficientnet-b1 -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 1 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6  # VRAM 6116-2093MB (pre)
#python ModelTrainer.py -n efficientnet-b2_size_10 -device 0 -b efficientnet-b2 -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 1 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6  # VRAM 6118-2093MB (pre)

#May 30
#python ModelTrainer.py -n efficientnet-b4_size_10 -device 0 -b efficientnet-b4 -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 16 -batch_fine 1 -size_sup 10 -size_qry 10 -num_qry 6  # VRAM 6118-2093MB (pre)
#python ModelTrainer.py -n efficientnet-b5_size_10 -device 0 -b efficientnet-b5 -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 16 -batch_fine 1 -size_sup 10 -size_qry 10 -num_qry 6  # VRAM 6118-2093MB (pre)

#python ModelTrainer.py -n resnet50      -device 0 -b resnet50      -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 36 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6  # VRAM 6118-2093MB (pre)
#python ModelTrainer.py -n resnet50_swav_new -device 0 -b resnet50_swav -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 36 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6  # VRAM 6118-2093MB (pre)
#python ModelTrainer.py -n resnet50_qry_1      -device 0 -b resnet50        -aug_j 1 -epoch_pre 1 -epoch_fine 30 -batch_pre 128 -batch_fine 12 -size_sup 10 -size_qry 1 -num_qry 6
#python ModelTrainer.py -n efficientnet-b0_new -device 0 -b efficientnet-b0 -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 16 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6
#python ModelTrainer.py -n efficientnet-b1_new -device 0 -b efficientnet-b1 -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 16 -batch_fine 2 -size_sup 10 -size_qry 10 -num_qry 6
#python ModelTrainer.py -n efficientnet-b2_new -device 0 -b efficientnet-b2 -aug_j 1 -epoch_pre 3 -epoch_fine 30 -batch_pre 16 -batch_fine 2 -size_sup 10 -size_qry 10 -num_qry 6

#python ModelTrainer.py -n 4locations      -device 0 -b resnet50   -aug_r 1 -aug_j 1 -aug_d 1 -epoch_pre 3 -epoch_fine 60 -batch_pre 36 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6

#python ModelTrainer.py -n 15locations      -device 0 -b resnet50   -aug_r 1 -aug_j 1 -aug_d 1 -epoch_pre 3 -epoch_fine 40 -batch_pre 36 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6
python ModelTrainer.py -n 15locations_pre15      -device 0 -b resnet50   -aug_r 1 -aug_j 1 -aug_d 1 -epoch_pre 15 -epoch_fine 40 -batch_pre 36 -batch_fine 3 -size_sup 10 -size_qry 10 -num_qry 6

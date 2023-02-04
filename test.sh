#!/bin/bash

#python testing.py -n base
#
#python testing.py -n r
#python testing.py -n j
#python testing.py -n d

#python testing.py -n rj
#python testing.py -n rd
#python testing.py -n jd
#
#python testing.py -n rjd

#python testing.py -n efficientnet-b0_size_10 -d 3 -s 10
#python testing.py -n efficientnet-b1_size_10 -d 3 -s 10
#python testing.py -n efficientnet-b2_size_10 -d 3 -s 10

#python testing.py -n size_1 -d 3 -s 1
#python testing.py -n size_3 -d 3 -s 3

#python testing.py -n efficientnet-b4_size_10 -d 3 -s 10

#May 30
#python testing.py -n resnet50      -d 3 -s 10
#python testing.py -n resnet50_swav -d 3 -s 10

#May 31
#python testing.py -n resnet50      -d 3 -s 1
#python testing.py -n resnet50_swav_new    -d 3 -s 10
#python testing.py -n resnet50_qry_1       -d 3 -s 1
#python testing.py -n efficientnet-b0_new  -d 3 -s 10
#python testing.py -n efficientnet-b1_new  -d 3 -s 10
#python testing.py -n efficientnet-b2_new  -d 3 -s 10

#Dec 6
#python testing.py -n f1_base -d 1 -p /mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/av/ckpt/dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes/model_best_dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes.pth
#python testing.py -n f1_skip_cov -d 1 -p /mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/av/ckpt/dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_skip_cov_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes/model_best_dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_skip_cov_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes.pth
#python testing.py -n f1_swav -d 1 -p /mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/av/ckpt/dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_swav_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes/model_best_dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_swav_FineTune_NewMix_SymMah_batch3_10-shot_lr_1e-05_lrsch_0.5_10_100episodes.pth

#python testing.py -n f1_scratch -d 1 -p /mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/av/ckpt/dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_scratch_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes/ckpt_all/dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_scratch_FineTune_NewMix_SymMah_batch3_10-shot_lr_1e-05_lrsch_0.5_10_100episodes_epoch_10.pth
#python testing.py -n f1_skip_cov -d 1 -p /mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/av/ckpt/dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_skip_cov_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes/model_best_dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_skip_cov_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes.pth
#python testing.py -n f1_swav_skip_cov -d 1 -p /mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/av/ckpt/dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_swav_skip_cov_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes/model_best_dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_swav_skip_cov_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes.pth

#python testing.py -n f1_skip_cov -d 1 -p /mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/av/ckpt/dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_skip_cov_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes/model_best_dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_skip_cov_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes.pth

#Jan 18
python testing.py -n f1_swav_tr5 -d 1 -p /mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/av/ckpt/dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_swav_tr5_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes/model_best_dual_fisheye_exclude_Kemper3F_WestVillageStudyHall_EnvironmentalScience1F_batch_3_neg_50_15locations_pre15_val_per_landmark_f1_swav_tr5_batch36_10-shot_lr_0.0001_lrsch_0.5_10_16episodes.pth

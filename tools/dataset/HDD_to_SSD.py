import os
import os.path as osp
import shutil
from glob import glob


if __name__ == '__main__':
    src = '/mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/dataset/av/dual_cam/hdf5/new'
    dst = '/home/nick/dataset/dual_fisheye_indoor/PNG'

    locations = os.listdir(src)
    for location in locations:
        dir_dst_cw = osp.join(dst, location, 'cw')
        dir_dst_ccw = osp.join(dst, location, 'ccw')
        for dir_direction_dst in [dir_dst_cw, dir_dst_ccw]:
            if not osp.exists(dir_direction_dst):
                os.makedirs(dir_direction_dst)

        dir_location_src = osp.join(src, location)
        laps = [l for l in os.listdir(dir_location_src) if '.hdf5' not in l and 'mav' in l]
        for lap in laps:
            dir_lap_src = osp.join(dir_location_src, lap)
            if 'ccw' in lap or 'CCW' in lap:
                dir_direction_dst = dir_dst_ccw
            else:
                dir_direction_dst = dir_dst_cw
            dir_lap_dst = osp.join(dir_direction_dst, lap)
            print('Copying from {} to {}'.format(dir_lap_src, dir_lap_dst))
            shutil.copytree(dir_lap_src, dir_lap_dst)

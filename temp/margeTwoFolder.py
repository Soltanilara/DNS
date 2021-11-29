import os
import os.path as osp
import glob
import shutil


dir_sources = ['/Users/shidebo/dataset/AV/hfd5/ASB1F/210830_180728_mav_320x240ASB1F',
               '/Users/shidebo/dataset/AV/hfd5/ASB1F/210830_181657_mav_320x240ASB1F']
dir_target = '/Users/shidebo/dataset/AV/merged/ASB1F/test_flatten'

if __name__ == '__main__':
    paths = []
    for d in dir_sources:
        paths += glob.glob(d + '/*.jpg')

    for i, path in enumerate(paths):
        shutil.copy2(path, osp.join(dir_target, str(i).zfill(5)+'.jpg'))

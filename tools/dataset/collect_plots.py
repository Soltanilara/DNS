import os
import os.path as osp
from glob import glob
import shutil
from tqdm import tqdm


if __name__ == '__main__':
    dir_dataset = '/home/nick/dataset/dual_fisheye_indoor/PNG'
    dir_output = '/home/nick/dataset/dual_fisheye_indoor/PNG_plots'
    path_plots = glob(osp.join(dir_dataset, '*', '*', '*', '*.html'))

    for path_plot in tqdm(path_plots):
        lap = path_plot.split('/')[-2]
        path_dst = osp.join(dir_output, lap+'.html')
        shutil.copy(path_plot, path_dst)

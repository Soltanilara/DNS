import os
import os.path as osp
from glob import glob
import shutil


def checkLapNumber(path_txt):
    dic = {}
    for path in path_txt:
        splitted = path.split(osp.sep)
        location = splitted[6]
        direction = splitted[7]

        if location not in dic:
            dic[location] = {}
        if direction not in dic[location]:
            dic[location][direction] = []

        dic[location][direction].append(path)
    pass


if __name__ == '__main__':
    dir_dataset = '/home/nick/dataset/dual_fisheye_indoor/PNG'
    dir_output = '/home/nick/dataset/dual_fisheye_indoor/landmark'

    path_txt = glob(osp.join(dir_dataset, '*', 'cw', '*', 'Analysis', 'landmarks.txt'))
    path_txt += glob(osp.join(dir_dataset, '*', 'ccw', '*', 'Analysis', 'landmarks.txt'))

    # check if the total number of laps is an even number
    checkLapNumber(path_txt)

    for path in path_txt:
        lap_name = path.split(osp.sep)[8] + '.txt'
        path_output = osp.join(dir_output, lap_name)
        shutil.copy(path, path_output)

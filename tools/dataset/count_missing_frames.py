import os
import os.path as osp
from glob import glob


def check_lap(dir_lap):
    cnt_missing = 0
    categories = os.listdir(dir_lap)
    for cat in categories:
        dir_cat = osp.join(dir_lap, cat)
    path_imgs = sorted(glob(osp.join(dir_cat, '*.png')))
    ids = [int(p.split('/')[-1][:-4]) for p in path_imgs]
    for i in range(len(ids) - 2):
        cnt_missing += (ids[i + 1] - ids[i] - 1)
    print(f'{lap}: {cnt_missing}')


if __name__ == '__main__':
    dir_dataset = '/path/to/Dual Fisheye'

    locations = os.listdir(dir_dataset)

    for location in locations:
        dir_location = osp.join(dir_dataset, location)
        print(f'------------{location}------------')
        for direction in ['cw', 'ccw']:
            laps = [l for l in os.listdir(osp.join(dir_location, direction)) if '.' not in l]
            for lap in laps:
                dir_lap = osp.join(dir_dataset, location, direction, lap)
                cnt_missing = 0
                categories = [cat for cat in os.listdir(dir_lap) if '_' in cat and '.' not in cat]
                for cat in categories:
                    dir_cat = osp.join(dir_lap, cat)
                    path_imgs = sorted(glob(osp.join(dir_cat, '*.png')))
                    ids = [int(p.split('/')[-1][:-4]) for p in path_imgs]
                    for i in range(len(ids)-2):
                        cat_missing = ids[i+1] - ids[i] - 1
                        if cat_missing > 0:
                            cnt_missing += cat_missing
                print(f'{lap}: {cnt_missing}')

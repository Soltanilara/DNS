import sys
import os
import os.path as osp
import shutil

import h5py
import numpy as np
from PIL import Image, ImageOps
from glob import glob


class Counter:
    def __init__(self, dir_lap):
        self.dir_lap = dir_lap

        # For 3 classes
        self.cnt_left = 0
        self.cnt_straight = 0
        self.cnt_right = 0

    def add_left(self):
        self.cnt_left += 1

    def add_straight(self):
        self.cnt_straight += 1

    def add_right(self):
        self.cnt_right += 1

    def check_dir(self, dir_dst):
        if not osp.exists(dir_dst):
            os.makedirs(dir_dst)
        return dir_dst

    def get_dir_left(self):
        dir_dst = osp.join(self.dir_lap, 'left', str(self.cnt_left).zfill(3))
        return self.check_dir(dir_dst)

    def get_dir_straight(self):
        dir_dst = osp.join(self.dir_lap, 'straight', str(self.cnt_straight).zfill(3))
        return self.check_dir(dir_dst)

    def get_dir_right(self):
        dir_dst = osp.join(self.dir_lap, 'right', str(self.cnt_right).zfill(3))
        return self.check_dir(dir_dst)


class Counter_FSL(Counter):
    def __init__(self, dir_lap):
        super(Counter_FSL, self).__init__(dir_lap)
        self.cnt = 0

    def get_id(self):
        digit_0 = self.cnt // 20
        digit_1 = (self.cnt - digit_0 * 20) // 2
        digit_2 = self.cnt % 2
        id = '{}{}{}'.format(digit_0, digit_1, digit_2)
        return id

    def get_dir_negative(self):
        dir_dst = osp.join(self.dir_lap, '{}_negative'.format(self.get_id()))
        self.cnt += 1
        return self.check_dir(dir_dst)

    def get_dir_left(self):
        dir_dst = osp.join(self.dir_lap, '{}_left'.format(self.get_id()))
        self.cnt += 1
        return self.check_dir(dir_dst)

    def get_dir_straight(self):
        dir_dst = osp.join(self.dir_lap, '{}_straight'.format(self.get_id()))
        self.cnt += 1
        return self.check_dir(dir_dst)

    def get_dir_right(self):
        dir_dst = osp.join(self.dir_lap, '{}_right'.format(str(self.get_id()).zfill(3)))
        self.cnt += 1
        return self.check_dir(dir_dst)


def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, key)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, key in h5py_dataset_iterator(hdf_file):
        yield path, key


def is_dual_camera(hdf_file):
    for group in hdf_file.keys():
        keys = [key for key in hdf_file[group].keys()]
        if 'imgLeft' in keys and 'imgRight' in keys:
            return True
        else:
            return False


def crop_fisheye(arr):
    return arr[:, 30:290, :]


def get_landmark_to_hdf5(dir_landmarks, dir_hdf5):
    path_landmark = glob(osp.join(dir_landmarks, '*.txt'))
    path_landmark.sort()
    path_hdf5 = glob(osp.join(dir_hdf5, '*', '*', '*.hdf5')) + glob(osp.join(dir_hdf5, '*', '*.hdf5'))

    dict_landmark_to_hdf5 = {}
    for landmark in path_landmark:
        lap_id = osp.basename(landmark)[:13]
        for hdf5 in path_hdf5:
            if lap_id in hdf5:
                dict_landmark_to_hdf5[landmark] = hdf5
                break
            if hdf5 == path_hdf5[-1]:
                pass
    return dict_landmark_to_hdf5


def read_landmark(path_landmark):
    with open(path_landmark, 'r') as f:
        landmarks = f.readlines()
    landmarks = [eval(i) for i in landmarks]
    return landmarks


def transfer_img(i_start, i_end, dir_src, dir_dst):
    for i in range(i_start, i_end):
        name_img = str(i).zfill(5) + '.png'
        path_src = osp.join(dir_src, name_img)
        path_dst = osp.join(dir_dst, name_img)
        shutil.move(path_src, path_dst)


if __name__ == '__main__':
    # DEBUG = True
    DEBUG = False

    if sys.platform == 'linux':
        dir_landmarks = '/volume1/dataset/av/dual_fisheye/landmarks/'
        dir_hdf5 = '/volume1/homes/SDB/Box/Autonomous Driving/Campus Data/HDF5'
        dir_output = '/volume1/dataset/av/dual_fisheye/sorted/PNG'
    else:
        dir_landmarks = '/Users/shidebo/dataset/AV/landmarks'
        dir_hdf5 = '/Volumes/home/Box/Autonomous Driving/Campus Data/HDF5'
        dir_output = '/Users/shidebo/dataset/AV/3Classes'

    if DEBUG:
        dict_landmark_to_hdf5 = {
            '/Users/shidebo/dataset/AV/landmarks/220311_212128_mav_320x240_Ghausi2F_Lounge_CCW.txt': '/Volumes/home/Box/Autonomous Driving/Campus Data/HDF5/Ghausi2F_Lounge/ccw/220311_212128_mav_320x240_Ghausi2F_Lounge_CCW.hdf5'}
    else:
        dict_landmark_to_hdf5 = get_landmark_to_hdf5(dir_landmarks, dir_hdf5)

    ## -------------------------- Extract Images and Steering Data -----------------------
    for i, (path_landmark, path_hdf5) in enumerate(dict_landmark_to_hdf5.items()):
        print('[{}/{}] Working on {}'.format(i, len(dict_landmark_to_hdf5), osp.basename(path_landmark)[:-4]))

        location = path_landmark.split('_')[-2]
        if location == 'Lounge':
            location = '_'.join(path_landmark.split('_')[-3:-1])
        lap = osp.basename(path_landmark)[:-4]
        dir_output_lap = osp.join(dir_output, location, lap)
        if osp.exists(dir_output_lap):
            shutil.rmtree(dir_output_lap)
        os.makedirs(dir_output_lap)

        if location in ['switched', 'Lounge']:
            location = path_landmark.split('_')[-3]

        h = []
        num_img = 0
        foundImg = 0
        foundSteering = 0
        dict = {}
        with h5py.File(path_hdf5, 'r') as f:
            DUAL_CAM = is_dual_camera(f)
            for dset, key in traverse_datasets(f):
                if f[dset].shape == (240, 320, 3):
                    if DUAL_CAM:
                        if key == 'imgLeft':
                            j = np.array(f[dset][:])
                            array_l = np.reshape(j, (240, 320, 3))
                            array_l = crop_fisheye(array_l)
                            SAVE = False
                        elif key == 'imgRight':
                            j = np.array(f[dset][:])
                            array_r = np.reshape(j, (240, 320, 3))
                            array_r = crop_fisheye(array_r)
                            if 'switched' not in path_hdf5:
                                array = np.concatenate((array_l, array_r), axis=1)
                            else:
                                array = np.concatenate((array_r, array_l), axis=1)
                            SAVE = True

                    else:
                        SAVE = True
                        j = np.array(f[dset][:])
                        array = np.reshape(j, (240, 320, 3))
                    if SAVE:
                        if DEBUG:
                            im = Image.fromarray(np.zeros((1, 1, 3), dtype='uint8'))
                        else:
                            im = Image.fromarray(array)
                            im = ImageOps.flip(im)
                        file_name = str(num_img)
                        file_name = file_name.zfill(5)
                        file_name = osp.join(dir_output_lap, file_name+".png")
                        if not osp.exists(file_name):
                            im.save(file_name)
                        num_img = num_img + 1
                        foundImg = 1
                if 'steering' in dset:
                    h.append(np.array(f[dset]))
                    dict[num_img - 1] = np.array(f[dset])
                    foundSteering = 1
                if foundImg == 1 and foundSteering == 1:
                    foundImg = 0
                    foundSteering = 0

        ## ----------------------------------- Moving Images into Folders -----------------------------
        # For 3 classes
        # for cat in ['left', 'straight', 'right']:
        #     dir_to_make = osp.join(lap, cat)
        #     if not osp.exists(dir_to_make):
        #         os.makedirs(dir_to_make)

        landmarks = read_landmark(path_landmark)

        start = 0
        # counter = Counter(dir_output_lap)
        counter = Counter_FSL(dir_output_lap)
        if 'ccw' in path_landmark or 'CCW' in path_landmark:
            get_dir_turn = counter.get_dir_left
        else:
            get_dir_turn = counter.get_dir_right

        for i, end in enumerate(landmarks):

            # For 3 classes
            # if i % 2 == 0:
            #     dir_dst = counter.get_dir_straight()
            #     counter.add_straight()
            # elif 'ccw' in path_landmark or 'CCW' in path_landmark:
            #     dir_dst = counter.get_dir_left()
            #     counter.add_left()
            # else:
            #     dir_dst = counter.get_dir_right()
            #     counter.add_right()
            # transfer_img(start, end, dir_output_lap, dir_dst)
            # start = end

            # For FSL
            dir_dst = counter.get_dir_negative()
            end -= 15
            transfer_img(start, max(start, end), dir_output_lap, dir_dst)

            if start >= end:
                print('[Warning] Might have index error. start: {}, end: {}'.format(start, end))

            start = end
            end = start + 15
            if i % 2 == 0:
                dir_dst = get_dir_turn()
            else:
                dir_dst = counter.get_dir_straight()
            transfer_img(start, end, dir_output_lap, dir_dst)

            if start >= end:
                print('[Warning] Might have index error. start: {}, end: {}'.format(start, end))

            start = end
        dir_dst = counter.get_dir_negative()
        transfer_img(start, num_img, dir_output_lap, dir_dst)

import os.path as osp
import h5py
from glob import glob


def check_hdf5(path):
    lap = osp.basename(path)
    print(f'Checking lap: {lap}')
    h5 = h5py.File(path, "r+")
    id_last = -1

    for frame in h5.keys():
        id = int(frame.split('_')[-1])
        if id == id_last + 1 \
                and h5[frame]['imgLeft'].shape == h5[frame]['imgRight'].shape == (240, 320, 3) \
                and h5[frame]['steering'].size == 1 \
                and h5[frame]['throttle'].size == 1:
            pass
        else:
            print('Error frame: {}'.format(id))
        id_last = id


if __name__ == '__main__':
    # dir_hdf5 = r'/Volumes/SDB/HDF5/test_0123'
    # dir_hdf5 = r'/mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/dataset/av/dual_cam/hdf5/Walker2F'
    # dir_hdf5 = r'/mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/dataset/av/dual_cam/hdf5/new/EngineeringLibrary'
    # paths_hdf5 = glob(osp.join(dir_hdf5, '*.hdf5'))

    # paths_hdf5 = ['/mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/dataset/av/dual_cam/hdf5/Walker2F/220719_162811_mav_320x240CCW.hdf5']
    paths_hdf5 = ['/mnt/18ee5ff4-5aaf-495c-b305-9b9698c8d053/Nick/dataset/av/dual_cam/hdf5/new/EngineeringLibrary/220810_161241_mav_320x240cw.hdf5']

    for path_hdf5 in paths_hdf5:
        check_hdf5(path_hdf5)

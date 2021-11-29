import os
import os.path as osp
import shutil


def divide_folders(root):
    cw = []
    ccw = []
    for d in root:
        if d.endswith('ccw'):
            if osp.basename(d) == 'ccw':
                ccw = [osp.join(d, subDir) for subDir in os.listdir(d) if subDir != '.DS_Store']
            else:
                ccw.append(d)
        else:
            if osp.basename(d) == 'cw':
                cw = [osp.join(d, subDir) for subDir in os.listdir(d) if subDir != '.DS_Store']
            else:
                cw.append(d)
    cw.sort()
    ccw.sort()
    return cw, ccw


def isCWCCWAsFolder(dirs):
    basenames = [osp.basename(d) for d in dirs]
    if 'cw' in basenames and 'ccw' in basenames:
        return True
    return False


dir_root = r'/Users/shidebo/dataset/AV/Sorted/ASB1F'
dir_target = r'/Users/shidebo/dataset/AV/merged/ASB1F'
num_pairs_train = 2
num_pairs_val = 1
num_pairs_test = 1

if __name__ == '__main__':
    folders = [osp.join(dir_root, f) for f in os.listdir(dir_root) if not f.startswith('.')]
    folders_cw, folders_ccw = divide_folders(folders)
    folders_pairs = [[cw, ccw] for cw, ccw in zip(folders_cw, folders_ccw)]
    folders_train = folders_pairs[: num_pairs_train]
    folders_val = folders_pairs[num_pairs_train: num_pairs_train + num_pairs_val]
    folders_test = folders_pairs[num_pairs_train + num_pairs_val: num_pairs_train + num_pairs_val + num_pairs_test]

    for target, folders_source in zip(['train', 'val', 'test'], [folders_train, folders_val, folders_test]):
        folder_to_img = {}
        for folders_cwccw in folders_source:
            for folder_direction in folders_cwccw:
                # direction = folder_direction.split('_')[-1]
                direction = folder_direction.split('/')[-2]
                subfolders = [f for f in os.listdir(folder_direction) if f not in [
                    '.DS_Store', 'plot.png', 'plot.html', 'plot.json', 'Analysis']]
                for subfolder in subfolders:
                    if direction == 'cw':
                        subfolder_target = subfolder
                    else:
                        idx_new = str(int(subfolder[: 2]) + 8).zfill(2)
                        subfolder_target = idx_new + subfolder[2:]
                    dir_subfolder = osp.join(folder_direction, subfolder)
                    path_imgs = [osp.join(dir_subfolder, fname) for fname in os.listdir(dir_subfolder)]
                    if subfolder_target not in folder_to_img.keys():
                        folder_to_img[subfolder_target] = path_imgs
                    else:
                        folder_to_img[subfolder_target] += path_imgs
        dir_output = osp.join(dir_target, target)
        for subfolder, path_imgs in folder_to_img.items():
            dir_subfolder = osp.join(dir_output, subfolder)
            os.makedirs(dir_subfolder, exist_ok=True)
            path_imgs.sort()
            for i, path_img in enumerate(path_imgs):
                shutil.copy2(path_img, osp.join(dir_subfolder, str(i).zfill(5) + '.jpg'))

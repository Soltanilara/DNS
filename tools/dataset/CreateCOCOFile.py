import json
import os
import os.path as osp
from glob import glob


def save_json(path, info, images, annotations, categories):
    if not osp.exists(osp.dirname(path)):
        os.makedirs(osp.dirname(path))
    with open(path, 'w') as f:
        json.dump({
            'info': info,
            'images': images,
            'annotations': annotations,
            'categories': categories,
        }, f)


if __name__ == '__main__':
    # dataset_root = r'/Volumes/dataset/autonomous-navifation/sorted'
    # dir_output = r'/Users/shidebo/SynologyDrive/Projects/AV/code/coco'

    # dataset_root = r'/volume1/dataset/av/sorted'
    # dir_output = r'/var/services/homes/SDB/Drive/Projects/AV/code/coco/coco_indoor_exclude_Bainer2F_Kemper3F'

    dataset_root = r'/home/nick/dataset/dual_fisheye_indoor'
    dir_output = r'/home/nick/projects/FSL/coco/dual_fisheye/test/change_lap'

    # exclude_locations = [
    #     'Bainer2F', 'Kemper3F'
    # ]
    exclude_locations = [
        'ASB1F', 'ASB2F', 'Bainer2F', 'Ghausi2F_Lounge', 'Kemper3F'
    ]
    # exclude_locations = []

    info = {
        'description': 'Exclude:' + str(exclude_locations),
        'directions': ['cw', 'ccw'],
        'num_landmark': 8,
        'img_shape': [320, 240],
        'num_cats': 0
    }

    # for dataset_type in ['train', 'val', 'test', 'landmark']:
    for dataset_type in ['test', 'landmark']:
    # for dataset_type in ['train', 'val']:
        cat_id = -1
        img_id = -1
        ann_id = -1
        images = []
        annotations = []
        categories = []
        for dir_location in [i for i in glob(osp.join(dataset_root, '*')) if osp.isdir(i) and osp.basename(i) not in exclude_locations]:
            if dataset_type in ['test', 'landmark']:
                cat_id = -1
                img_id = -1
                ann_id = -1
                images = []
                annotations = []
                categories = []
            location = osp.basename(dir_location)
            num_laps = len([i for i in glob(osp.join(dir_location, '*', '*')) if osp.basename(i) != '@eaDir'])
            num_val = 2 if num_laps > 8 else 1
            num_test = 1
            num_train = int(num_laps / 2) - num_val - num_test
            for direction in info['directions']:
                landmark_list = []
                dir_direction = osp.join(dataset_root, location, direction)
                laps = [i for i in os.listdir(dir_direction) if osp.isdir(osp.join(dir_direction, i)) and i != '@eaDir']
                laps.sort()
                if dataset_type == 'train':
                    laps_type = laps[:num_train]
                elif dataset_type == 'val':
                    laps_type = laps[num_train:num_train+num_val]
                elif dataset_type == 'test':
                    laps_type = laps[num_train+num_val:]
                else:
                    laps_type = laps[num_train + num_val - 1]
                    # laps_type = laps[-3]
                    if not isinstance(laps_type, list):
                        laps_type = [laps_type]

                print('Creating dataset: {}, location: {}, direction: {}, num_lap: {}'.format(dataset_type, location, direction, len(laps_type)))

                for lap in laps_type:
                    dir_lap = osp.join(dir_direction, lap)
                    landmarks = [i for i in os.listdir(dir_lap) if osp.isdir(osp.join(dir_lap, i))]
                    landmarks.sort()
                    for landmark in landmarks:
                        dir_landmark = osp.join(dir_lap, landmark)
                        if len(glob(osp.join(dir_landmark, '*.jpg'))) > 0:
                            cat_id += 1
                            cat = {
                                'location': location,
                                'direction': direction,
                                'lap': lap,
                                'id': cat_id,
                                'name': landmark
                            }
                            categories.append(cat)
                            info['num_cats'] += 1
                        path_imgs = glob(osp.join(dir_landmark, '*.jpg'))
                        path_imgs.sort()
                        for path_img in path_imgs:
                            fname = osp.basename(path_img)
                            img_id += 1
                            img = {
                                'file_name': osp.join(location, direction, lap, landmark, fname),
                                'file_path': osp.join(location, direction, lap, landmark, fname),
                                'id': img_id,
                            }
                            images.append(img)

                            ann_id += 1
                            ann = {
                                'image_id': img_id,
                                'category_id': cat_id,
                                'id': ann_id,
                            }
                            annotations.append(ann)
            if dataset_type == 'test':
                save_json(osp.join(dir_output, dataset_type + '_' + location + '.json'), info, images, annotations, categories)
            elif dataset_type == 'landmark':
                save_json(osp.join(dir_output, 'test_' + location + '_landmark' + '.json'), info, images, annotations, categories)

        if dataset_type not in ['test', 'landmark']:
            save_json(osp.join(dir_output, dataset_type + '.json'), info, images, annotations, categories)

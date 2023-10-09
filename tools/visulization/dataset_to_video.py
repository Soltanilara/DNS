import cv2
import os
import os.path as osp
from glob import glob


if __name__ == '__main__':
    dir_dataset = '/path/to/Dual Fisheye'
    dir_output = 'MP4_new'
    locations = ['ASB1F', 'WestVillageStudyHall', 'EnvironmentalScience1F']

    res = (260, 120)
    fps = 25
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    for location in locations:
        dir_location = osp.join(dir_dataset, location)
        for direction in ['cw', 'ccw']:
            laps = [l for l in os.listdir(osp.join(dir_location, direction)) if '.' not in l]
            laps = sorted(laps)
            for lap in laps:
                dir_lap = osp.join(dir_dataset, location, direction, lap)
                path_imgs = sorted(glob(osp.join(dir_lap, '0*', '*.png')))
                dir_video = osp.join(dir_output, location, direction)
                if not osp.exists(dir_video):
                    os.makedirs(dir_video)
                path_video = osp.join(dir_output, location, direction, f'{lap}.mp4')
                out = cv2.VideoWriter(path_video, fourcc, fps, res)
                for path_img in path_imgs:
                    category, filename = path_img.split('/')[-2:]
                    img = cv2.imread(path_img)
                    img = cv2.resize(img, res)
                    img = cv2.putText(
                        img,
                        f'{category}/{filename}',
                        org=(0, 12),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255),
                        thickness=1
                    )
                    out.write(img)
                out.release()
                print(f'Done: {path_video}')

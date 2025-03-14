import os
import shutil
import cv2
from tqdm import tqdm

depth_dirs = [
    r'/AkhmetzyanovD/projects/hztfm/dataset/data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02',
    r'/AkhmetzyanovD/projects/hztfm/dataset/data_depth_annotated/train/2011_09_26_drive_0009_sync/proj_depth/groundtruth/image_02',
    r'/AkhmetzyanovD/projects/hztfm/dataset/data_depth_annotated/train/2011_09_26_drive_0011_sync/proj_depth/groundtruth/image_02',
    r'/AkhmetzyanovD/projects/hztfm/dataset/data_depth_annotated/val/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02',
    r'/AkhmetzyanovD/projects/hztfm/dataset/data_depth_annotated/val/2011_09_26_drive_0005_sync/proj_depth/groundtruth/image_02',
    r'/AkhmetzyanovD/projects/hztfm/dataset/data_depth_annotated/val/2011_09_26_drive_0013_sync/proj_depth/groundtruth/image_02',
    r'/AkhmetzyanovD/projects/hztfm/dataset/data_depth_annotated/train/2011_09_26_drive_0014_sync/proj_depth/groundtruth/image_02',
    r'/AkhmetzyanovD/projects/hztfm/dataset/data_depth_annotated/train/2011_09_26_drive_0017_sync/proj_depth/groundtruth/image_02',    
    r'/AkhmetzyanovD/projects/hztfm/dataset/data_depth_annotated/train/2011_09_26_drive_0018_sync/proj_depth/groundtruth/image_02'
]

img_dirs = [
    r'/AkhmetzyanovD/projects/hztfm/dataset/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_02/data',
    r'/AkhmetzyanovD/projects/hztfm/dataset/2011_09_26_drive_0009_sync/2011_09_26/2011_09_26_drive_0009_sync/image_02/data',
    r'/AkhmetzyanovD/projects/hztfm/dataset/2011_09_26_drive_0011_sync/2011_09_26/2011_09_26_drive_0011_sync/image_02/data',
    r'/AkhmetzyanovD/projects/hztfm/dataset/2011_09_26_drive_0002_sync/2011_09_26/2011_09_26_drive_0002_sync/image_02/data',
    r'/AkhmetzyanovD/projects/hztfm/dataset/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_02/data',
    r'/AkhmetzyanovD/projects/hztfm/dataset/2011_09_26_drive_0013_sync/2011_09_26/2011_09_26_drive_0013_sync/image_02/data',
    r'/AkhmetzyanovD/projects/hztfm/dataset/2011_09_26_drive_0014_sync/2011_09_26/2011_09_26_drive_0014_sync/image_02/data',
    r'/AkhmetzyanovD/projects/hztfm/dataset/2011_09_26_drive_0017_sync/2011_09_26/2011_09_26_drive_0017_sync/image_02/data',
    r'/AkhmetzyanovD/projects/hztfm/dataset/2011_09_26_drive_0018_sync/2011_09_26/2011_09_26_drive_0018_sync/image_02/data'
]

prefixs = ['0001', '0009', '0011', '0002', '0005', '0013', '0014', '0017', '0018']

save_dir = r'/AkhmetzyanovD/projects/hztfm/dataset/data_depth_v2'

train = []
valid = []

def get_real_height(depth):
    for index in range(len(depth)):
        if sum(depth[index, :]) != 0:
            return index

for prefix, depth_dir, img_dir in tqdm(zip(prefixs, depth_dirs, img_dirs)):
    for i, imgname in enumerate(sorted(os.listdir(depth_dir))):
        imgfile = os.path.join(img_dir, imgname)
        depthfile = os.path.join(depth_dir, imgname)

        if i < len(os.listdir(depth_dir)) * 0.8:
            img_dst_dir = os.path.join(save_dir, 'train', 'images')
            depth_dst_dir = os.path.join(save_dir, 'train', 'depths')
        else:
            img_dst_dir = os.path.join(save_dir, 'valid', 'images')
            depth_dst_dir = os.path.join(save_dir, 'valid', 'depths')

        if not os.path.exists(img_dst_dir):
            os.makedirs(img_dst_dir)
        if not os.path.exists(depth_dst_dir):
            os.makedirs(depth_dst_dir)

        img_dst = os.path.join(img_dst_dir, f'{prefix}_{imgname}')
        depth_dst = os.path.join(depth_dst_dir, f'{prefix}_{imgname}')

        if img_dst.split('/')[-3] == 'train':
            train.append([img_dst, depth_dst])
        else:
            valid.append([img_dst, depth_dst])
        
        shutil.copy(imgfile, img_dst)
        shutil.copy(depthfile, depth_dst)
        # image = cv2.imread(imgfile)
        # depth = cv2.imread(depthfile)[:, :, 0]
        # index = get_real_height(depth)
        # if index + 5 < len(depth):
        #     image = image[index + 5:, :]
        #     depth = depth[index + 5:, :]
        # else:
        #     image = image[index:, :]
        #     depth = depth[index:, :]

        # cv2.imwrite(img_dst, image)
        # cv2.imwrite(depth_dst, depth)


data = {'train': train, 'valid': valid}

for part in ['train', 'valid']:
    with open(f'{save_dir}/{part}.txt', 'w') as file:
        for line in data[part]:
            file.write(f'{line[0]} {line[1]}\n')

        file.close()

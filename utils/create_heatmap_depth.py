import matplotlib.pyplot as plt
import cv2
depth_path = r'/AkhmetzyanovD/projects/hztfm/dataset/data_depth_annotated/train/2011_09_26_drive_0009_sync/proj_depth/groundtruth/image_02/0000000005.png'
depth = cv2.imread(depth_path)[:, :, 0]
save_path = '/AkhmetzyanovD/projects/hztfm/depth_pipeline/test/pictures/depth.png'
plt.imshow(depth, cmap='inferno')
plt.axis('off')
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
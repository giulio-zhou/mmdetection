import glob
import numpy as np
import os
import pandas as pd
import skimage.io as skio
from .custom import CustomDataset

class DistillDataset(CustomDataset):
    CLASSES = ('vehicle',)
    def load_annotations(self, ann_file):
        self.labels, self.img_infos = [], []
        label_df = pd.read_csv(ann_file)
        video_dirs = sorted(glob.glob(self.img_prefix + '/*'))
        img_paths = map(lambda x: sorted(glob.glob(x + '/*')), video_dirs)
        for video_dir, video_img_paths in zip(video_dirs, img_paths):
            video_df = label_df[label_df['filename'] == video_dir]
            height, width = skio.imread(video_img_paths[0]).shape[:2]
            for frame_no, img_path in enumerate(video_img_paths):
                frame_gt = video_df[video_df['frame_no'] == frame_no]
                if len(frame_gt) > 0:
                    gt_bboxes = frame_gt[['xmin', 'ymin', 'xmax', 'ymax']]
                    gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                    gt_bboxes *= [width, height, width, height]
                    # Assume only one category for now.
                    gt_labels = np.ones(len(gt_bboxes), dtype=np.int64)
                    if 'conf' in frame_gt.columns:
                        distill_targets = np.array(frame_gt['conf'])
                    else:
                        distill_targets = gt_labels
                else:
                    gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                    gt_labels = np.array([], dtype=np.int64)
                    distill_targets = np.array([], dtype=np.float32)
                ann = dict(bboxes=gt_bboxes, labels=gt_labels,
                           distill_targets=distill_targets,
                           bboxes_ignore=np.zeros((0, 4), dtype=np.float32))
                img_info = dict(filename=img_path, height=height, width=width)
                self.labels.append(ann)
                self.img_infos.append(img_info)
        self.img_ids = [i for i in range(len(self.labels))]
        self.cat_ids = [i for i in range(len(DistillDataset.CLASSES))]
        # For now, use img_prefix once and set it to empty afterwards.
        self.img_prefix = ''
        return self.img_infos
    def get_ann_info(self, idx):
        return self.labels[idx]
    def _filter_imgs(self, min_size=32):
        valid_inds = filter(lambda i: len(self.labels[i]) > 0,
                            np.arange(len(self.labels)))
        return valid_inds

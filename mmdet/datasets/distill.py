import glob
import json
import mmcv
import numpy as np
import os
import os.path as osp
import pandas as pd
import skimage.io as skio

from .custom import CustomDataset
from .extra_aug import ExtraAugmentation
from .utils import to_tensor, random_scale
from mmcv.parallel import DataContainer as DC

class DistillDataset(CustomDataset):
    """Custom dataset for detection, with support for distillation.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4),
                'labels_ignore': <np.ndarray> (k, 4), (optional field)
                'distill_targets': <np.ndarray> (n, c) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """
    CLASSES = ('vehicle',)
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 **kwargs):
        self.ann_files = ann_file.split(',')
        if 'loss_weights' in kwargs:
            self.loss_weights = kwargs['loss_weights']
            del kwargs['loss_weights']
        else:
            self.loss_weights = np.ones(len(self.ann_files))
        self.loss_weights = np.array(self.loss_weights, dtype=np.float32)
        assert len(self.loss_weights) == len(self.ann_files)
        super(DistillDataset, self).__init__(ann_file, img_prefix,
                                             img_scale, img_norm_cfg, **kwargs)

    def _load_annotations(self, df):
        labels, img_infos = [], []
        video_dirs = sorted(glob.glob(self.img_prefix + '/*'))
        img_paths = map(lambda x: sorted(glob.glob(x + '/*')), video_dirs)
        for video_dir, video_img_paths in zip(video_dirs, img_paths):
            video_df = df[df['filename'] == video_dir]
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
                labels.append(ann)
                img_infos.append(img_info)
        return labels, img_infos

    def load_annotations(self, ann_files):
        self.coco, self.labels, self.img_infos = None, [], []
        print(self.img_prefix)
        print(ann_files)
        for ann_path in ann_files.split(','):
            label_df = pd.read_csv(ann_path)
            labels, img_infos = self._load_annotations(label_df)
            self.labels.append(labels)
            self.img_infos.append(img_infos)
            # Create COCO annotation object.
            if (self.coco is None) and ('conf' not in label_df.columns):
                self.coco = self._get_ann_file(ann_path)
        self.labels = list(zip(*self.labels))
        self.img_infos = self.img_infos[0]
        self.img_ids = [i for i in range(len(self.img_infos))]
        self.cat_ids = [i for i in range(len(DistillDataset.CLASSES))]
        # For now, use img_prefix once and set it to empty afterwards.
        self.img_prefix = ''
        return self.img_infos
    def get_ann_info(self, idx):
        # Assume the first element contains the relevant annotations.
        return self.labels[idx][0]
    def _filter_imgs(self, min_size=32):
        valid_inds = []
        for i in range(len(self.labels)):
            if sum([len(x['bboxes']) for x in self.labels[i]]) > 0:
                valid_inds.append(i)
        # Pre-filter img_ids and labels as well for consistency.
        self.img_ids = [self.img_ids[i] for i in valid_inds]
        self.labels = [self.labels[i] for i in valid_inds]
        return valid_inds
    def _get_ann_file(self, label_path, default_path='distill_anno.json'):
        if os.path.exists(default_path):
            return default_path
        df = pd.read_csv(label_path)
        filenames = sorted(np.unique(df['filename']))
        num_annos, i = 0, 0
        json_file = {}
        json_file['categories'] = [dict(id=0, name='obj', supercategory='obj')]
        json_file['images'], json_file['annotations'] = [], []
        for filename in filenames:
            print(filename)
            img_paths = sorted(glob.glob(filename + '/*'))
            height, width = skio.imread(img_paths[0]).shape[:2]
            for frame_no in range(len(img_paths)):
                frame_df = df[(df['filename'] == filename) &
                              (df['frame_no'] == frame_no)]
                json_file['images'].extend(
                  [{'id': i,
                    'width': width,
                    'height': height,
                    'file_name': img_paths[frame_no]}])
                if len(frame_df) == 0:
                    i += 1
                    continue
                dets = np.array(frame_df[['xmin', 'ymin', 'xmax', 'ymax']])
                dets *= [width, height, width, height]
                xs = dets[:, 0]
                ys = dets[:, 1]
                ws = dets[:, 2] - xs # + 1
                hs = dets[:, 3] - ys # + 1
                anno_ids = [j for j in range(num_annos,
                                             num_annos + dets.shape[0])]
                xs, ys, ws, hs = list(map(lambda x: list(map(float, x)),
                                          [xs, ys, ws, hs]))
                json_file['annotations'].extend(
                  [{'id': anno_ids[k], 'image_id' : i,
                    'category_id' : 0, # only one category
                    'bbox' : [xs[k], ys[k], ws[k], hs[k]],
                    'area': ws[k]*hs[k],
                    'iscrowd': 0} for k in range(dets.shape[0])])
                i += 1
                num_annos += dets.shape[0]
        with open(default_path, 'w') as outfile:
            json.dump(json_file, outfile)
            return default_path
    def _set_group_flag(self):
        """Set flag according to whether labels are ensemble predictions,
        or actual ground truth.

        Images with ground truth labels will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            ann = self.labels[i][0]
            distill_targets = ann['distill_targets']
            if distill_targets.dtype == np.int64:
                self.flag[i] = 1
        print(np.where(self.flag)[0])
    def prepare_train_img(self, idx):
        """Override function to prepare train inputs to allow multiple
        distillation targets.
        """
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        # TODO: Ignore masks and bboxes_ignore for now.
        # Join together bboxes, labels, bboxes_ignore, distill_targets, masks
        # for augmentations, keep indices to resplit after-the-fact.
        ann = self.labels[idx]
        ann_indices, gt_bboxes, gt_labels, distill_targets = [], [], [], []
        for i in range(len(ann)):
            num_bboxes = len(ann[i]['labels'])
            ann_indices.extend([i] * num_bboxes)
            gt_bboxes.extend(ann[i]['bboxes'])
            gt_labels.extend(ann[i]['labels'])
            distill_targets.extend(ann[i]['distill_targets'])
        ann_indices, gt_bboxes, gt_labels = (
            np.array(x) for x in [ann_indices, gt_bboxes, gt_labels])
        distill_targets = np.array(distill_targets, dtype=np.float64)
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, idx = self.extra_aug(img, gt_bboxes,
                                                 np.arange(len(gt_labels)))
            gt_labels = gt_labels[idx]
            distill_targets = distill_targets[idx]
            ann_indices = ann_indices[idx]

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales)  # sample a scale
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        def split_fn(elems, idx):
            output = []
            for i in range(len(ann)):
                matching_elems = elems[np.where(idx == i)]
                output.append(DC(to_tensor(matching_elems)))
            return output
        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=split_fn(gt_bboxes, ann_indices))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = split_fn(gt_labels, ann_indices)
        if self.with_crowd:
            data['gt_bboxes_ignore'] = split_fn(gt_bboxes_ignore, ann_indices)
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if True:
            data['distill_targets'] = split_fn(distill_targets, ann_indices)
            data['loss_weights'] = DC(to_tensor(self.loss_weights))
        return data

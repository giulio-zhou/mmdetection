import glob
import json
import mmcv
import numpy as np
import os
import os.path as osp
import pandas as pd
import skimage.io as skio
from PIL import Image

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
    COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 **kwargs):
        self.ann_files = ann_file.split(',')
        # Assign CLASSES.
        if 'categories' in kwargs:
            if kwargs.get('categories', None) == 'coco':
                self.CLASSES = self.COCO_CLASSES
            del kwargs['categories']
        # Handle default args in kwargs.
        self.balanced = kwargs.get('balanced', False)
        if 'balanced' in kwargs:
            self.balanced[1] = np.array(self.balanced[1])
            self.balanced[1] /= np.sum(self.balanced[1])
            print(self.balanced)
            del kwargs['balanced']
        self.subsample = kwargs.get('subsample', False)
        if 'subsample' in kwargs:
            del kwargs['subsample']
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
            for frame_no, img_path in enumerate(video_img_paths):
                width, height = Image.open(img_path).size
                frame_gt = video_df[video_df['frame_no'] == frame_no]
                if len(frame_gt) > 0:
                    gt_bboxes = frame_gt[['xmin', 'ymin', 'xmax', 'ymax']]
                    gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                    gt_bboxes *= [width, height, width, height]
                    # Map detection category to index.
                    if 'category' in frame_gt.columns:
                        gt_labels = [self.CLASSES.index(i) for i in frame_gt['category']]
                        gt_labels = np.array(gt_labels, dtype=np.int64) + 1
                    else:
                        gt_labels = np.ones(len(gt_bboxes), dtype=np.int64)
                    if 'conf' in frame_gt.columns:
                        # Fill all other columns with uniform 1 - conf.
                        confs = np.array(frame_gt['conf'])
                        # residual_confs = (1 - confs) / len(CLASSES)
                        distill_targets = np.zeros((len(gt_bboxes), len(self.CLASSES)),
                                                   dtype=np.float32)
                        # distill_targets[:] = residual_confs[:, None]
                        distill_targets[(np.arange(len(gt_bboxes)), gt_labels - 1)] = confs
                    else:
                        distill_targets = gt_labels
                else:
                    # Instead of throwing away images with no bounding boxes, 
                    # add a size 0 bounding box at (0, 0) that won't be matched.
                    gt_bboxes = 0.5 * np.array([[width, height, width, height]],
                                               dtype=np.float32)
                    gt_labels = np.array([0], dtype=np.int64)
                    distill_targets = np.zeros((1, len(self.CLASSES)), dtype=np.float32)
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
        print(len(self.img_infos))
        self.img_ids = [i for i in range(len(self.img_infos))]
        self.cat_ids = [i for i in range(len(self.CLASSES))]
        # For now, use img_prefix once and set it to empty afterwards.
        self.img_prefix = ''
        return self.img_infos
    def get_ann_info(self, idx):
        # Assume the first element contains the relevant annotations.
        return self.labels[idx][0]
    def _filter_imgs(self, min_size=32):
        keep = np.ones(len(self.labels), dtype=np.uint8)
        print(self.subsample)
        if self.subsample:
            num_samples = int(len(self.labels) * self.subsample['sample_frac'])
            if self.subsample['mode'] == 'uniform':
                idx = np.linspace(0, len(self.labels), num_samples,
                                  endpoint=False, dtype=np.int32)
                keep[idx] = 0
            keep = 1 - keep
        valid_inds = np.where(keep)[0]
        # for i in range(len(self.labels)):
            # if keep[i] and sum([len(x['bboxes']) for x in self.labels[i]]) > 0:
            #     valid_inds.append(i)
        print(valid_inds)
        # Pre-filter img_ids and labels as well for consistency.
        self.img_ids = [self.img_ids[i] for i in valid_inds]
        self.labels = [self.labels[i] for i in valid_inds]
        return valid_inds
    def _get_ann_file(self, label_path, default_path='distill_anno.json'):
        if os.path.exists(default_path):
            return default_path
        df = pd.read_csv(label_path)
        # filenames = sorted(np.unique(df['filename']))
        filenames = sorted(glob.glob(self.img_prefix + '/*'))
        num_annos, i = 0, 0
        json_file = {}
        json_file['categories'] = [dict(id=i, name=c, supercategory='obj')
                                   for i, c in enumerate(self.CLASSES)]
        json_file['images'], json_file['annotations'] = [], []
        for filename in filenames:
            print(filename)
            img_paths = sorted(glob.glob(filename + '/*'))
            for frame_no in range(len(img_paths)):
                frame_df = df[(df['filename'] == filename) &
                              (df['frame_no'] == frame_no)]
                width, height = Image.open(img_paths[frame_no]).size
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
                if 'category' in df.columns:
                    cats = [self.CLASSES.index(j) for j in np.array(frame_df['category'])]
                else:
                    cats = [0 for _ in range(dets.shape[0])]
                json_file['annotations'].extend(
                  [{'id': anno_ids[k], 'image_id' : i,
                    'category_id' : cats[k],
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

        In balanced mode, Images with ground truth labels will be set as
        group 0, otherwise group 1.
        """
        self.flag = np.ones(len(self), dtype=np.uint8)
        if self.balanced:
            sampling_frac = self.balanced[0]
            elems_to_sample = int(sampling_frac * len(self))
            # Assuming the first element is ground truth.
            valid_idx = np.array([i for i in range(len(self))
                                  if len(self.labels[i][0]['bboxes']) > 0])
            samples = np.linspace(0, len(valid_idx), elems_to_sample,
                                  endpoint=False, dtype=np.int32)
            self.flag[valid_idx[samples]] = 0
        print(0, np.where(self.flag == 0)[0])
        print(1, np.where(self.flag == 1)[0])
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
        distill_targets = np.array(distill_targets)
        # print('h', np.mean(distill_targets), distill_targets.shape)
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, i = self.extra_aug(img, gt_bboxes,
                                               np.arange(len(gt_labels)))
            gt_labels = gt_labels[i]
            distill_targets = distill_targets[i]
            ann_indices = ann_indices[i]

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
        if self.balanced:
            # Mutually exclusive weighting.
            # loss_weights = np.eye(2, dtype=np.float32)[self.flag[idx]]
            # data['loss_weights'] = DC(to_tensor(loss_weights))
            data['gt_bboxes'][1-self.flag[idx]] = DC(to_tensor([]))
            data['loss_weights'] = DC(to_tensor(np.ones(len(ann))))
        else:
            data['loss_weights'] = DC(to_tensor(self.loss_weights))
        return data

from .single_stage import SingleStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class RetinaNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)


@DETECTORS.register_module
class RetinaDistillNet(RetinaNet):
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, distill_targets):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, distill_targets, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(*loss_inputs)
        return losses

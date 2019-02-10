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
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, distill_targets, loss_weights):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_weights = loss_weights[0] # Same for each example.
        losses_list = []
        for i in range(len(gt_bboxes)):
            loss_inputs = outs + (gt_bboxes[i], gt_labels[i], distill_targets[i],
                                  img_metas, self.train_cfg)
            losses_list.append(self.bbox_head.loss(*loss_inputs))
        losses = {}
        for key in losses_list[0]:
            weighted_losses = []
            for i, loss in enumerate(losses_list):
                weighted_losses += map(lambda x: x * loss_weights[i], loss[key])
            losses[key] = weighted_losses
        return losses

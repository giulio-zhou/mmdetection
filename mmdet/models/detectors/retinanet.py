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
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      distill_targets, loss_weights):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        losses = {}
        # There is guaranteed to be at least one input that has bboxes.
        for i in range(len(gt_bboxes)):
            idx = [j for j in range(len(gt_bboxes[i]))
                   if gt_bboxes[i][j].size()[0] > 0]
            if len(idx) == 0:
                continue
            curr_loss_weights = [loss_weights[j][i] for j in idx]
            curr_outs = tuple([[outs[k][n][idx] for n in range(len(outs[k]))] for k in range(len(outs))])
            loss_inputs = curr_outs + ([gt_bboxes[i][j] for j in idx],
                                       [gt_labels[i][j] for j in idx],
                                       [distill_targets[i][j] for j in idx],
                                       [img_metas[j] for j in idx], self.train_cfg)
            loss_dict = self.bbox_head.loss(*loss_inputs)
            for key, loss_list in loss_dict.items():
                weighted_loss = [l * w for l, w in zip(loss_list, curr_loss_weights)]
                losses[key] = losses.get(key, []) + weighted_loss
        return losses

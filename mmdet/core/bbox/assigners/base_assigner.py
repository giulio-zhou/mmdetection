from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class BaseAssigner():

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        pass

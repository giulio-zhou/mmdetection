import argparse
import glob
import os

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors
from tensorboardX import SummaryWriter


def capture_stdout(run_fn, temp_path='temp123.txt'):
    import sys
    stdout_ = sys.stdout
    sys.stdout = open(temp_path, 'w')
    run_fn()
    sys.stdout = stdout_
    return open(temp_path, 'r').read()


def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file or checkpoint dir')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if os.path.isdir(args.checkpoint):
        print(args.checkpoint)
        checkpoints = glob.glob(args.checkpoint + '/epoch_*.pth')
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('epoch_')[-1].split('.')[0]))
        print(checkpoints)
        if args.out is not None:
            if not os.path.exists(args.out):
                os.mkdir(args.out)
            elif os.path.isfile(args.out):
                raise ValueError('args.out must be a directory.')
        # Create TensorBoard writer for output checkpoint dir.
        tensorboard_writer = SummaryWriter(args.out)
    else:
        checkpoints = [args.checkpoint]
        tensorboard_writer = None
        if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
            raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    for i, checkpoint in enumerate(checkpoints):
        outpath = args.out
        if os.path.isdir(args.checkpoint):
            outpath = args.out + '/%d_out.pkl' % i

        if not os.path.exists(outpath):
            if args.gpus == 1:
                model = build_detector(
                    cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
                load_checkpoint(model, checkpoint)
                model = MMDataParallel(model, device_ids=[0])

                data_loader = build_dataloader(
                    dataset,
                    imgs_per_gpu=1,
                    workers_per_gpu=cfg.data.workers_per_gpu,
                    num_gpus=1,
                    dist=False,
                    shuffle=False)
                outputs = single_test(model, data_loader, args.show)
            else:
                model_args = cfg.model.copy()
                model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
                model_type = getattr(detectors, model_args.pop('type'))
                outputs = parallel_test(
                    model_type,
                    model_args,
                    checkpoint,
                    dataset,
                    _data_func,
                    range(args.gpus),
                    workers_per_gpu=args.proc_per_gpu)

        # TODO: Currently assume test set is same size as training set.
        num_iters = (i+1) * len(dataset)
        if outpath:
            if os.path.exists(outpath):
                print('reading results from {}'.format(outpath))
                outputs = mmcv.load(outpath)
            else:
                print('writing results to {}'.format(outpath))
                mmcv.dump(outputs, outpath)
            eval_types = args.eval
            if eval_types:
                print('Starting evaluate {}'.format(' and '.join(eval_types)))
                if eval_types == ['proposal_fast']:
                    result_file = outpath
                    coco_eval(result_file, eval_types, dataset.coco)
                else:
                    if not isinstance(outputs[0], dict):
                        result_file = outpath + '.json'
                        results2json(dataset, outputs, result_file)
                        results_dict = coco_eval(result_file, eval_types, dataset.coco)
                        if tensorboard_writer:
                            for eval_type in eval_types:
                                out = capture_stdout(lambda: results_dict[eval_type].summarize())
                                for line in out.split('\n')[:-1]:
                                    parts = line.split('=')
                                    name, score = '='.join(parts[:-1]), float(parts[-1])
                                    tensorboard_writer.add_scalar('eval/' + name, score, num_iters)
                    else:
                        for name in outputs[0]:
                            print('\nEvaluating {}'.format(name))
                            outputs_ = [out[name] for out in outputs]
                            result_file = outpath + '.{}.json'.format(name)
                            results2json(dataset, outputs_, result_file)
                            results_dict = coco_eval(result_file, eval_types, dataset.coco)
                            if tensorboard_writer:
                                for eval_type in eval_types:
                                    out = capture_stdout(lambda: results_dict[eval_type].summarize())


if __name__ == '__main__':
    main()

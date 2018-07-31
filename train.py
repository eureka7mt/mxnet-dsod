import argparse
import mxnet as mx
import os
import sys
from train_net import train_net


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
    parser.add_argument('--train-path', dest='train_path', help='train record to use',
                        default=os.path.join(os.getcwd(), 'data', 'train.rec'), type=str)
    parser.add_argument('--val-path', dest='val_path', help='validation record to use',
                        default=os.path.join(os.getcwd(), 'data', 'val.rec'), type=str)
    parser.add_argument('--network', dest='network', type=str, default='dsod',
                        help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=8,
                        help='training batch size')
    parser.add_argument('--accum-batch-size', dest='accum_batch_size', type=int, default=128,
                        help='training batch size')                        
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    parser.add_argument('--epochs', dest='epochs', help='number of epochs of training',
                        default=100000, type=int)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
                        help='set image shape')
    parser.add_argument('--label-width', dest='label_width', type=int, default=350,
                        help='force padding label width to sync across train and validation')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='sgd',
                        help='Whether to use a different optimizer or follow the original code with sgd')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0.0005,
                        help='weight decay')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--lr-steps', dest='lr_refactor_step', type=list, default=[20000, 40000, 60000, 80000],
                        help='refactor learning rate at specified epochs')
    parser.add_argument('--lr-factor', dest='lr_refactor_ratio', type=int, default=0.1,
                        help='ratio to refactor learning rate')
    parser.add_argument('--num-class', dest='num_class', type=int, default=20,
                        help='number of classes')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='aeroplane, bicycle, bird, boat, bottle, bus, \
                        car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.45,
                        help='non-maximum suppression threshold')
    parser.add_argument('--nms_topk', dest='nms_topk', type=int, default=400,
                        help='final number of detections')
    parser.add_argument('--overlap', dest='overlap_thresh', type=float, default=0.5,
                        help='evaluation overlap threshold')
    parser.add_argument('--use-difficult', dest='use_difficult', type=bool, default=False,
                        help='use difficult ground-truths in evaluation')
    parser.add_argument('--voc07', dest='use_voc07_metric', type=bool, default=True,
                        help='use PASCAL VOC 07 11-point metric')
    parser.add_argument('--min_neg_ratio', dest='min_neg_ratio', type=int, default=3,
                        help='min ratio of negative samples taken in hard mining.')

    args = parser.parse_args()
    return args

def parse_class_names(args):
    """ parse # classes and class_names if applicable """
    num_class = args.num_class
    if len(args.class_names) > 0:
        if os.path.isfile(args.class_names):
            # try to open it to read class names
            with open(args.class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in args.class_names.split(',')]
        assert len(class_names) == num_class, str(len(class_names))
        for name in class_names:
            assert len(name) > 0
    else:
        class_names = None
    return class_names

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    # class names if applicable
    class_names = parse_class_names(args)
    # start training
    train_net(args.network, args.train_path, ctx, args.accum_batch_size,
                  args.num_class, args.batch_size,
                  args.data_shape, [args.mean_r, args.mean_g, args.mean_b],
                  args.epochs, args.learning_rate, args.momentum, args.weight_decay,
                  args.lr_refactor_step, args.lr_refactor_ratio,
                  val_path=args.val_path,
                  min_neg_ratio=args.min_neg_ratio,
                  class_names=class_names,
                  label_pad_width=args.label_width,
                  nms_thresh=args.nms_thresh,
                  nms_topk=args.nms_topk,
                  ovp_thresh=args.overlap_thresh,
                  use_difficult=args.use_difficult,
                  voc07_metric=args.use_voc07_metric,
                  optimizer=args.optimizer
                  )
import mxnet as mx
import numpy as np
import time
import sys
import os
import metric
import codecs
from mxnet import nd
from mxnet import init
from mxnet import gluon
from mxnet import autograd
from mxnet.contrib.ndarray import MultiBoxDetection
from mxnet.contrib.ndarray import MultiBoxPrior
from mxnet.contrib.ndarray import MultiBoxTarget
from mxnet.gluon import nn
from net import DSOD300
from grpnet import Gate_layer
from iterator import DetRecordIter
from config.config import cfg
from evaluate.eval_metric import MApMetric, VOC07MApMetric

def class_predictor(num):
    return nn.Conv2D(num, 3, padding=1)

def box_predictor(num_anchors):
    return nn.Conv2D(num_anchors*4, 3 , padding=1)

def flatten_prediction(pred):
    return nd.flatten(nd.transpose(pred, axes=(0,2,3,1)))

def concat_prediction(preds):
    return nd.concat(*preds, dim=1)

def training_targets(anchors, class_preds, labels, min_neg_ratio):
    class_preds=class_preds.transpose(axes=(0, 2, 1))
    return MultiBoxTarget(anchors, labels, class_preds, negative_mining_ratio=min_neg_ratio)

def validate(net, val_data, ctx, valid_metric, clip=False):
    val_data.reset()
    valid_metric.reset()
    for i, batch in enumerate(val_data):
        x = batch.data[0].as_in_context(ctx)
        y = batch.label
        box_preds, cls_preds, anchors = net(x)
        cls_probs = nd.SoftmaxActivation(cls_preds.transpose((0,2,1)), mode='channel')
        z = MultiBoxDetection(cls_probs, box_preds, anchors, force_suppress=False, clip=clip, nms_threshold=0.45, nms_topk=400)
        valid_metric.update(y, z)
    return 0

class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        pj = output.pick(label, axis=self._axis, keepdims=True)
        loss = -self._alpha*((1-pj)**self._gamma)*pj.log()
        return loss.mean(axis=self._batch_axis, exclude=True)

class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output-label)*mask, scalar=1.0)
        return loss.mean(self._batch_axis, exclude=True)

def get_optimizer_params(optimizer=None, learning_rate=None, momentum=None,
                         weight_decay=None, ctx=None):
    if optimizer.lower() == 'rmsprop':
        opt = 'rmsprop'
        print('you chose RMSProp, decreasing lr by a factor of 10')
        optimizer_params = {'learning_rate': learning_rate / 10.0,
                            'wd': weight_decay,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    elif optimizer.lower() == 'sgd':
        opt = 'sgd'
        optimizer_params = {'learning_rate': learning_rate,
                            'momentum': momentum,
                            'wd': weight_decay,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    elif optimizer.lower() == 'adadelta':
        opt = 'adadelta'
        optimizer_params = {}
    elif optimizer.lower() == 'adam':
        opt = 'adam'
        optimizer_params = {'learning_rate': learning_rate,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    return opt, optimizer_params

def multibox_layer_model(body, num_classes, sizes, ratios, num_predictions_layer=6):
    # create location prediction layer
    # create class prediction layer
    box_predictors = nn.HybridSequential()
    class_predictors = nn.HybridSequential()
    
    for i in range(num_predictions_layer):
        size = sizes[i]
        ratio = ratios[i]
        num_anchors = len(size) -1 + len(ratio)
        num_cls_pred = num_anchors * num_classes
        box_predictors.add(box_predictor(num_anchors))
        class_predictors.add(class_predictor(num_cls_pred))

    model = nn.HybridSequential()
    model.add(body, class_predictors, box_predictors)
    return model


def multibox_layer_forward(x, model, num_classes, sizes=[.2, .95],
                    ratios=[1], normalization=-1, num_channels=[],
                    clip=False, interm_layer=0, steps=[],  verbose=False):
    """
    the basic aggregation module for SSD detection. Takes in multiple layers,
    generate multiple object detection targets by customized layers
    Parameters:
    ----------
    from_layers : list
        generate multibox detection from layers
    num_classes : int
        number of classes excluding background, will automatically handle
        background in this function
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    num_channels : list of int
        number of input layer channels, used when normalization is enabled, the
        length of list should equals to number of normalization layers
    clip : bool
        whether to clip out-of-image boxes
    interm_layer : int
        if > 0, will add a intermediate Convolution layer
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    Returns:
    ----------
    list of outputs, as [loc_preds, cls_preds, anchor_boxes]
    loc_preds : localization regression prediction
    cls_preds : classification prediction
    anchor_boxes : generated anchor boxes
    """
    body, class_predictors, box_predictors = model
    from_layers = body(x)
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 0, \
        "num_classes {} must be larger than 0".format(num_classes)

    assert len(ratios) > 0, "aspect ratios must not be empty list"
    if not isinstance(ratios[0], list):
        # provided only one ratio list, broadcast to all from_layers
        ratios = [ratios] * len(from_layers)
    assert len(ratios) == len(from_layers), \
        "ratios and from_layers must have same length"

    assert len(sizes) > 0, "sizes must not be empty list"
    assert len(sizes) == len(from_layers), \
        "sizes and from_layers must have same length"

    if not isinstance(normalization, list):
        normalization = [normalization] * len(from_layers)
    assert len(normalization) == len(from_layers)
    assert sum(x > 0 for x in normalization) <= len(num_channels), \
        "must provide number of channels for each normalized layer"

    if steps:
        assert len(steps) == len(from_layers), "provide steps for all layers or leave empty"

    box_pred_layers = []
    class_pred_layers = []
    anchor_layers = []

    for k, from_layer in enumerate(from_layers):
        # normalize
        if normalization[k] > 0:
            from_layer = nd.L2Normalization(data=from_layer, \
                mode="channel")
            scale = nd.full(shape=(1, num_channels[k], 1, 1), val=normalization[k])
            from_layer = nd.broadcast_mul(lhs=scale, rhs=from_layer)
        #if interm_layer > 0:
        #    from_layer = mx.symbol.Convolution(data=from_layer, kernel=(3,3), \
        #        stride=(1,1), pad=(1,1), num_filter=interm_layer, \
        #        name="{}_inter_conv".format(from_name))
        #    from_layer = mx.symbol.Activation(data=from_layer, act_type="relu", \
        #        name="{}_inter_relu".format(from_name))

        size = sizes[k]
        assert len(size) > 0, "must provide at least one size"
        size_str = "(" + ",".join([str(x) for x in size]) + ")"
        ratio = ratios[k]
        assert len(ratio) > 0, "must provide at least one ratio"
        ratio_str = "(" + ",".join([str(x) for x in ratio]) + ")"

        box_pred_layers.append(flatten_prediction(box_predictors[k](from_layer)))

        class_pred_layers.append(flatten_prediction(class_predictors[k](from_layer)))

        # create anchor generation layer
        if steps:
            step = (steps[k], steps[k])
        else:
            step = '(-1.0, -1.0)'
        anchors = MultiBoxPrior(from_layer, sizes=size_str, ratios=ratio_str, \
            clip=clip, steps=step)
        anchors = nd.Flatten(data=anchors)
        anchor_layers.append(anchors)

        if verbose:
            print('Predict scale', k, from_layer.shape, 'with',
                  anchor_layers[-1].shape[1], 'anchors')

    box_preds = nd.concat(*box_pred_layers, num_args=len(box_pred_layers), \
        dim=1, name="multibox_loc_pred")
    cls_preds = nd.concat(*class_pred_layers, num_args=len(class_pred_layers), \
        dim=1)
    cls_preds = cls_preds.reshape(shape=(0, -1, num_classes))
    anchor_boxes = nd.concat(*anchor_layers, num_args=len(anchor_layers), dim=1)
    anchor_boxes = anchor_boxes.reshape(shape=(0, -1, 4))
    return [box_preds, cls_preds, anchor_boxes]

def train_net(network, train_path, ctx, accum_batch_size, num_classes, batch_size,
            data_shape, mean_pixels, epochs, learning_rate, momentum, 
            weight_decay, lr_refactor_step, lr_refactor_ratio,
            val_path='', min_neg_ratio=0, class_names=[],
            label_pad_width=350, nms_thresh=0.45, nms_topk=400,
            ovp_thresh=0.5, use_difficult=False,
            voc07_metric=True, optimizer='sgd'):
    cls_metric = metric.Accuracy(ignore_label=-1)
    box_metric = metric.MAE()

    # optimizer
    opt, opt_params = get_optimizer_params(optimizer=optimizer, learning_rate=learning_rate, momentum=momentum,
                                           weight_decay=weight_decay, ctx=ctx)

    #load data
    ctx = ctx[0]
    data_shape = (3, data_shape, data_shape)
    train_list = ""
    val_list = ""
    train_iter = DetRecordIter(train_path, accum_batch_size, data_shape, mean_pixels=mean_pixels,
                               label_pad_width=label_pad_width, path_imglist=train_list, **cfg.train)
    #val_iter = DetRecordIter(val_path, accum_batch_size, data_shape, mean_pixels=mean_pixels,
    #                             label_pad_width=label_pad_width, path_imglist=val_list, **cfg.valid)
    val_iter = DetRecordIter(val_path, 32, data_shape, mean_pixels=mean_pixels,
                                 label_pad_width=label_pad_width, path_imglist=val_list, **cfg.valid)
    if network =='dsod':
        sizes = [[.1, .141], [.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1, 2, .5], [1, 2, .5, 3, 1./3], [1, 2, .5, 3, 1./3], [1, 2, .5, 3, 1./3], [1, 2, .5], [1, 2, .5]]
        normalization = [20,20,20,20,20,20]
        num_channels = [800, 512, 512, 256, 256, 256]
        clip = False
        interm_layer = 0
        steps = [x / 300.0 for x in [8, 16, 32, 64, 100, 300]]
        
        body = DSOD300(prefix = 'body_')
        body.initialize(init.Xavier, ctx=ctx)
        
        class Dsod(nn.HybridBlock):
            def __init__(self, num_classes, sizes, ratios, normalization, num_channels, interm_layer, steps, body, clip=False, verbose=False, **kwargs):
                super(Dsod, self).__init__(**kwargs)
                self.num_classes = num_classes + 1
                self.sizes = sizes
                self.ratios = ratios
                self.normalization = normalization
                self.num_channels = num_channels
                self.interm_layer = interm_layer
                self.steps = steps
                self.clip = clip
                self.verbose = verbose
                self.body = body
        
                with self.name_scope():
                    self.model = multibox_layer_model(self.body, self.num_classes, self.sizes, self.ratios)
            def hybrid_forward(self, F, x):
                box_preds, class_preds, anchors = multibox_layer_forward(x, self.model, self.num_classes, self.sizes, \
                    self.ratios, self.normalization, self.num_channels, \
                    self.clip, self.interm_layer, self.steps, self.verbose)
                return box_preds, class_preds, anchors
        
        net = Dsod(num_classes, sizes, ratios, normalization, num_channels, interm_layer, steps, body, prefix = 'pred_')
        net.initialize(init.Xavier(magnitude=2), ctx=ctx)
    elif network == 'grpdsod':
        sizes = [[.1, .141], [.2, .272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1, 1.6, 1./1.6, 2, .5, 3, 1./3], [1, 1.6, 1./1.6, 2, .5, 3, 1./3], [1, 1.6, 1./1.6, 2, .5, 3, 1./3], [1, 1.6, 1./1.6, 2, .5, 3, 1./3], [1, 1.6, 1./1.6, 2, .5], [1, 1.6, 1./1.6, 2, .5]]
        normalization = [20,20,20,20,20,20]
        num_channels = [1184, 513, 258, 258, 258, 256]
        clip = False
        interm_layer = 0
        steps = [x / 320.0 for x in [8, 16, 32, 64, 100, 320]]
        
        body = Gate_layer(channel_nums=num_channels, prefix = 'GRP_')
        body.initialize(init.Xavier(magnitude=3), ctx=mx.gpu(0))
        
        class GrpDsod(nn.HybridBlock):
            def __init__(self, num_classes, sizes, ratios, normalization, num_channels, interm_layer, steps, body, clip=False, verbose=False, **kwargs):
                super(GrpDsod, self).__init__(**kwargs)
                self.num_classes = num_classes + 1
                self.sizes = sizes
                self.ratios = ratios
                self.normalization = normalization
                self.num_channels = num_channels
                self.interm_layer = interm_layer
                self.steps = steps
                self.clip = clip
                self.verbose = verbose
                self.body = body
        
                with self.name_scope():
                    self.model = multibox_layer_model(self.body, self.num_classes, self.sizes, self.ratios)
            def hybrid_forward(self, F, x):
                box_preds, class_preds, anchors = multibox_layer_forward(x, self.model, self.num_classes, self.sizes, \
                    self.ratios, self.normalization, self.num_channels, \
                    self.clip, self.interm_layer, self.steps, self.verbose)
                return box_preds, class_preds, anchors       
        net = GrpDsod(num_classes, sizes, ratios, normalization, num_channels, interm_layer, steps, body, prefix = 'pred_')
        net.initialize(init.Xavier(magnitude=2), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), opt, opt_params)
    cls_loss = FocalLoss()
    box_loss = SmoothL1Loss()

    if voc07_metric:
        valid_metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names, pred_idx=3)
    else:
        valid_metric = MApMetric(ovp_thresh, use_difficult, class_names, pred_idx=3)

    iter_size = int(accum_batch_size/batch_size)
    #net.collect_params().reset_ctx(ctx)
    net.collect_params().setattr('grad_req', 'add')

    for epoch in range(epochs):
        train_iter.reset()
        cls_metric.reset()
        box_metric.reset()
        tic = time.time()
        if len(lr_refactor_step) > 0:
            if epoch == (lr_refactor_step[0]-1):
                trainer.set_learning_rate(learning_rate*lr_refactor_ratio)
                del lr_refactor_step[0]
        for i, batch in enumerate(train_iter):
            net.collect_params().zero_grad()
            for j in range(iter_size):
                x = batch.data[0][j*batch_size:(j+1)*batch_size].as_in_context(ctx)
                y = batch.label[0][j*batch_size:(j+1)*batch_size].as_in_context(ctx)
                with autograd.record():
                    box_preds, class_preds, anchors = net(x)
                    box_target, box_mask, cls_target = training_targets(anchors, class_preds, y, min_neg_ratio)
                    loss1 = cls_loss(class_preds, cls_target)
                    loss2 = box_loss(box_preds, box_target, box_mask)
                    loss = loss1 + loss2
                loss.backward()
                cls_metric.update([cls_target], [class_preds.transpose((0,2,1))])
                box_metric.update([box_target], [box_preds * box_mask])
    
            trainer.step(batch_size*iter_size)

        

        print('Epoch %2d, train %s %.2f, %s %.5f, time %.1f sec' % (epoch, *cls_metric.get(), *box_metric.get(), time.time()-tic))
        with codecs.open('%s_log'%network, 'a+', 'utf8') as f:
            f.write('Epoch %2d, train %s %.2f, %s %.5f, time %.1f sec' % (epoch, *cls_metric.get(), *box_metric.get(), time.time()-tic) + '\r\n')
        if (epoch+1) % 1000 == 0:
            validate(net, val_iter, ctx, valid_metric, clip=clip)
            names, values = valid_metric.get()
            for k in range(len(names)):
                print(names[k] + ':' + str(values[k]))
                with codecs.open('%s_log'%network, 'a+', 'utf8') as f:
                    f.write(names[k] + ':' + str(values[k]) + '\r\n')

        net.collect_params().save('model/Dsod_Epoch%d.params'%epoch)

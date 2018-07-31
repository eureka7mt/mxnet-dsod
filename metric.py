# coding: utf-8
# pylint: disable=no-member, too-many-lines
from __future__ import absolute_import
import math
from collections import OrderedDict

import numpy

from mxnet.base import numeric_types, string_types
from mxnet import ndarray
from mxnet import registry

def check_label_shapes(labels, preds, wrap=False, shape=False):
    """Helper function for checking shape of label and prediction
    Parameters
    ----------
    labels : list of `NDArray`
        The labels of the data.
    preds : list of `NDArray`
        Predicted values.
    wrap : boolean
        If True, wrap labels/preds in a list if they are single NDArray
    shape : boolean
        If True, check the shape of labels and preds;
        Otherwise only check their length.
    """
    if not shape:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

    if wrap:
        if isinstance(labels, ndarray.ndarray.NDArray):
            labels = [labels]
        if isinstance(preds, ndarray.ndarray.NDArray):
            preds = [preds]

    return labels, preds

class EvalMetric(object):
    """Base class for all evaluation metrics.
    .. note::
        This is a base class that provides common metric interfaces.
        One should not use this class directly, but instead create new metric
        classes that extend it.
    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self, name, output_names=None,
                 label_names=None, **kwargs):
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def get_config(self):
        """Save configurations of metric. Can be recreated
        from configs with metric.create(**config)
        """
        config = self._kwargs.copy()
        config.update({
            'metric': self.__class__.__name__,
            'name': self.name,
            'output_names': self.output_names,
            'label_names': self.label_names})
        return config

    def update_dict(self, label, pred):
        """Update the internal evaluation with named label and pred
        Parameters
        ----------
        labels : OrderedDict of str -> NDArray
            name to array mapping for labels.
        preds : OrderedDict of str -> NDArray
            name to array mapping of predicted outputs.
        """
        if self.output_names is not None:
            pred = [pred[name] for name in self.output_names]
        else:
            pred = list(pred.values())

        if self.label_names is not None:
            label = [label[name] for name in self.label_names]
        else:
            label = list(label.values())

        self.update(label, pred)

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.
        preds : list of `NDArray`
            Predicted values.
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_name_value(self):
        """Returns zipped name and value pairs.
        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

# pylint: disable=invalid-name
register = registry.get_register_func(EvalMetric, 'metric')
alias = registry.get_alias_func(EvalMetric, 'metric')
_create = registry.get_create_func(EvalMetric, 'metric')
# pylint: enable=invalid-name


def create(metric, *args, **kwargs):
    """Creates evaluation metric from metric names or instances of EvalMetric
    or a custom metric function.
    Parameters
    ----------
    metric : str or callable
        Specifies the metric to create.
        This argument must be one of the below:
        - Name of a metric.
        - An instance of `EvalMetric`.
        - A list, each element of which is a metric or a metric name.
        - An evaluation function that computes custom metric for a given batch of
          labels and predictions.
    *args : list
        Additional arguments to metric constructor.
        Only used when metric is str.
    **kwargs : dict
        Additional arguments to metric constructor.
        Only used when metric is str
    Examples
    --------
    >>> def custom_metric(label, pred):
    ...     return np.mean(np.abs(label - pred))
    ...
    >>> metric1 = mx.metric.create('acc')
    >>> metric2 = mx.metric.create(custom_metric)
    >>> metric3 = mx.metric.create([metric1, metric2, 'rmse'])
    """
    if callable(metric):
        return CustomMetric(metric, *args, **kwargs)
    elif isinstance(metric, list):
        composite_metric = CompositeEvalMetric()
        for child_metric in metric:
            composite_metric.add(create(child_metric, *args, **kwargs))
        return composite_metric

    return _create(metric, *args, **kwargs)

@register
@alias('composite')
class CompositeEvalMetric(EvalMetric):
    """Manages multiple evaluation metrics.
    Parameters
    ----------
    metrics : list of EvalMetric
        List of child metrics.
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> eval_metrics_1 = mx.metric.Accuracy()
    >>> eval_metrics_2 = mx.metric.F1()
    >>> eval_metrics = mx.metric.CompositeEvalMetric()
    >>> for child_metric in [eval_metrics_1, eval_metrics_2]:
    >>>     eval_metrics.add(child_metric)
    >>> eval_metrics.update(labels = labels, preds = predicts)
    >>> print eval_metrics.get()
    (['accuracy', 'f1'], [0.6666666666666666, 0.8])
    """

    def __init__(self, metrics=None, name='composite',
                 output_names=None, label_names=None):
        super(CompositeEvalMetric, self).__init__(
            name, output_names=output_names, label_names=label_names)
        if metrics is None:
            metrics = []
        self.metrics = [create(i) for i in metrics]

    def add(self, metric):
        """Adds a child metric.
        Parameters
        ----------
        metric
            A metric instance.
        """
        self.metrics.append(create(metric))

    def get_metric(self, index):
        """Returns a child metric.
        Parameters
        ----------
        index : int
            Index of child metric in the list of metrics.
        """
        try:
            return self.metrics[index]
        except IndexError:
            return ValueError("Metric index {} is out of range 0 and {}".format(
                index, len(self.metrics)))

    def update_dict(self, labels, preds): # pylint: disable=arguments-differ
        if self.label_names is not None:
            labels = OrderedDict([i for i in labels.items()
                                  if i[0] in self.label_names])
        if self.output_names is not None:
            preds = OrderedDict([i for i in preds.items()
                                 if i[0] in self.output_names])

        for metric in self.metrics:
            metric.update_dict(labels, preds)

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.
        preds : list of `NDArray`
            Predicted values.
        """
        for metric in self.metrics:
            metric.update(labels, preds)

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        try:
            for metric in self.metrics:
                metric.reset()
        except AttributeError:
            pass

    def get(self):
        """Returns the current evaluation result.
        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        names = []
        values = []
        for metric in self.metrics:
            name, value = metric.get()
            if isinstance(name, string_types):
                name = [name]
            if isinstance(value, numeric_types):
                value = [value]
            names.extend(name)
            values.extend(value)
        return (names, values)

    def get_config(self):
        config = super(CompositeEvalMetric, self).get_config()
        config.update({'metrics': [i.get_config() for i in self.metrics]})
        return config

@register
@alias('acci')
class Accuracy(EvalMetric):
    """Computes accuracy classification score.
    The accuracy score is defined as
    .. math::
        \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i=0}^{n-1}
        \\text{1}(\\hat{y_i} == y_i)
    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> acc = mx.metric.Accuracy()
    >>> acc.update(preds = predicts, labels = labels)
    >>> print acc.get()
    ('accuracy', 0.6666666666666666)
    """
    def __init__(self, ignore_label=-1, axis=1, name='accuracy',
                 output_names=None, label_names=None):
        super(Accuracy, self).__init__(
            name, ignore_label=ignore_label, axis=axis,
            output_names=output_names, label_names=label_names)
        self.axis = axis
        self.ignore_label = ignore_label

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data with class indices as values, one per sample.
        preds : list of `NDArray`
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')
            # flatten before checking shapes to avoid shape miss match
            label = label.flat
            pred_label = pred_label.flat

            check_label_shapes(label, pred_label) 
            for i in range(len(pred_label)):
                if label[i] != self.ignore_label:
                    self.num_inst += 1
                    if pred_label[i] == label[i]:
                        self.sum_metric += 1
                    else:
                        continue
                else:
                    continue


            #self.sum_metric += (pred_label == label).sum()
            #self.num_inst += len(pred_label)

####################
# REGRESSION METRICS
####################


@register
class MAE(EvalMetric):
    """Computes Mean Absolute Error (MAE) loss.
    The mean absolute error is given by
    .. math::
        \\frac{\\sum_i^n |y_i - \\hat{y}_i|}{n}
    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    Examples
    --------
    >>> predicts = [mx.nd.array(np.array([3, -0.5, 2, 7]).reshape(4,1))]
    >>> labels = [mx.nd.array(np.array([2.5, 0.0, 2, 8]).reshape(4,1))]
    >>> mean_absolute_error = mx.metric.MAE()
    >>> mean_absolute_error.update(labels = labels, preds = predicts)
    >>> print mean_absolute_error.get()
    ('mae', 0.5)
    """

    def __init__(self, name='mae',
                 output_names=None, label_names=None):
        super(MAE, self).__init__(
            name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.
        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)
            if len(pred.shape) == 1:
                pred = pred.reshape(pred.shape[0], 1)

            self.sum_metric += numpy.abs(label - pred).mean()
            self.num_inst += 1 # numpy.prod(label.shape)
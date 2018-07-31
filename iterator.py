import mxnet as mx
import numpy as np
import cv2

class DetRecordIter(mx.io.DataIter):
    """
    The new detection iterator wrapper for mx.io.ImageDetRecordIter which is
    written in C++, it takes record file as input and runs faster.
    Supports various augment operations for object detection.

    Parameters:
    -----------
    path_imgrec : str
        path to the record file
    path_imglist : str
        path to the list file to replace the labels in record
    batch_size : int
        batch size
    data_shape : tuple
        (3, height, width)
    label_width : int
        specify the label width, use -1 for variable length
    label_pad_width : int
        labels must have same shape in batches, use -1 for automatic estimation
        in each record, otherwise force padding to width in case you want t
        rain/validation to match the same width
    label_pad_value : float
        label padding value
    resize_mode : str
        force - resize to data_shape regardless of aspect ratio
        fit - try fit to data_shape preserving aspect ratio
        shrink - shrink to data_shape only, preserving aspect ratio
    mean_pixels : list or tuple
        mean values for red/green/blue
    kwargs : dict
        see mx.io.ImageDetRecordIter

    Returns:
    ----------

    """
    def __init__(self, path_imgrec, batch_size, data_shape, path_imglist="",
                 label_width=-1, label_pad_width=-1, label_pad_value=-1,
                 resize_mode='force',  mean_pixels=[123.68, 116.779, 103.939],
                 **kwargs):
        super(DetRecordIter, self).__init__()
        self.rec = mx.io.ImageDetRecordIter(
            path_imgrec     = path_imgrec,
            path_imglist    = path_imglist,
            label_width     = label_width,
            label_pad_width = label_pad_width,
            label_pad_value = label_pad_value,
            batch_size      = batch_size,
            data_shape      = data_shape,
            mean_r          = mean_pixels[0],
            mean_g          = mean_pixels[1],
            mean_b          = mean_pixels[2],
            resize_mode     = resize_mode,
            **kwargs)

        self.provide_label = None
        self._get_batch()
        if not self.provide_label:
            raise RuntimeError("Invalid ImageDetRecordIter: " + path_imgrec)
        self.reset()

    @property
    def provide_data(self):
        return self.rec.provide_data

    def reset(self):
        self.rec.reset()

    def iter_next(self):
        return self._get_batch()

    def next(self):
        if self.iter_next():
            return self._batch
        else:
            raise StopIteration

    def _get_batch(self):
        self._batch = self.rec.next()
        if not self._batch:
            return False

        if self.provide_label is None:
            # estimate the label shape for the first batch, always reshape to n*5
            first_label = self._batch.label[0][0].asnumpy()
            self.batch_size = self._batch.label[0].shape[0]
            self.label_header_width = int(first_label[4])
            self.label_object_width = int(first_label[5])
            assert self.label_object_width >= 5, "object width must >=5"
            self.label_start = 4 + self.label_header_width
            self.max_objects = (first_label.size - self.label_start) // self.label_object_width
            self.label_shape = (self.batch_size, self.max_objects, self.label_object_width)
            self.label_end = self.label_start + self.max_objects * self.label_object_width
            self.provide_label = [('label', self.label_shape)]

        # modify label
        label = self._batch.label[0].asnumpy()
        label = label[:, self.label_start:self.label_end].reshape(
            (self.batch_size, self.max_objects, self.label_object_width))
        self._batch.label = [mx.nd.array(label)]
        return True
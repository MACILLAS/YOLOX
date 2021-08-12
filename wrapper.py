#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import cv2
import numpy as np

import onnxruntime

from yolox.data.data_augment import preproc as preprocess

# from yolox.data.datasets import COCO_CLASSES
COCO_CLASSES = '0'
from yolox.utils import multiclass_nms, demo_postprocess


def open_sess(model='yolox.onnx'):
    return onnxruntime.InferenceSession(model)


def run(sess=None, img=None, input_shape="416,416", score=0.3):
    input_shape = tuple(map(int, input_shape.split(',')))
    origin_img = img
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img, ratio = preprocess(origin_img, input_shape, mean, std)
    session = sess

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]

    return final_boxes, final_scores, final_cls_inds

image = cv2.imread('./valid/00316_jpg.rf.ea1c6f25b0ad614226c16be8f0efe742.jpg')
session = open_sess(model='yolox.onnx')
final_boxes, final_scores, final_cls_inds = run(sess=session, img=image)

print(final_boxes)
print(final_scores)
print(final_cls_inds)
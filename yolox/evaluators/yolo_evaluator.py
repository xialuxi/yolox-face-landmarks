#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import sys
import tempfile
import time
from collections import ChainMap
from loguru import logger
from tqdm import tqdm

import numpy as np

import torch

from yolox.utils import gather, is_main_process, postprocess, synchronize, time_synchronized, get_local_rank
from yolox.data.datasets import YOLO_CLASSES

class YOLOEvaluator:
    """
    YOLO AP Evaluation class.
    """

    def __init__(
        self, dataloader, img_size, confthre, nmsthre, num_classes,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = len(YOLO_CLASSES)
        self.num_images = len(dataloader.dataset)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(
        self, model, distributed=False, half=False, trt_file=None, decoder=None, test_size=None
    ):
        """
        VOC average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO style AP of IoU=50:95
            ap50 (float) : VOC 2007 metric AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.HalfTensor if half else torch.FloatTensor
        if distributed:
            self.local_rank = get_local_rank()
            self.device = torch.device("cuda:{}".format(self.local_rank))
        #model = model.to(device)
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = {}
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        # for yolo test
        seen = 0
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
        nc = self.num_classes  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        # for yolo test

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                imgs = imgs.type(tensor_type).to(self.device)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)

                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )

                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end


                ##########  #####  #####  #####  #####  #####  #####  #####  #####  #####
                # Statistics per image
                for si, pred in enumerate(outputs):
                    img, gt_res, img_size, index = self.dataloader.dataset.pull_item(ids[si])
                    nl = len(gt_res)
                    gt_res = torch.from_numpy(np.array(gt_res)).to(self.device)
                    gt_bboxes, gt_cls = gt_res[:, 0:4], gt_res[:, 4]
                    seen += 1

                    if pred is None:
                        if nl:
                            stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), gt_cls.cpu()))
                        continue

                    pre_bboxes = pred[:, 0:4]
                    scale = min(self.img_size[0] / float(img_size[0]), self.img_size[1] / float(img_size[1]))
                    pre_bboxes /= scale
                    #pre_bboxes = xywh2xyxy(pre_bboxes)
                    pre_cls = pred[:, 6]
                    #pre_scores = pred[:, 4] * pred[:, 5]

                    # Assign all predictions as incorrect
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=self.device)
                    if nl:
                        detected = []  # target indices
                        tcls_tensor = gt_cls
                        # Per target class
                        for cls in torch.unique(tcls_tensor):
                            ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                            pi = (cls == pre_cls).nonzero(as_tuple=False).view(-1)  # target indices

                            if pi.shape[0]:
                                # Prediction to target ious
                                ious, i = self.box_iou(pre_bboxes[pi], gt_bboxes[ti]).max(1)  # best ious, indices

                                # Append detections
                                detected_set = set()
                                for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                    d = ti[i[j]]  # detected target
                                    if d.item() not in detected_set:
                                        detected_set.add(d.item())
                                        detected.append(d)
                                        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                        if len(detected) == nl:  # all targets already located in image
                                            break
                    # Append statistics (correct, conf, pcls, tcls)
                    stats.append((correct.cpu(), pred[:, 4].cpu(), pre_cls.cpu(), gt_cls.cpu()))

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        else:
            nt = torch.zeros(1)
        # Print results
        print(s)
        pf = '%20s' + '%12.3g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (YOLO_CLASSES[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        return map50, map, "test end and map is " + str(map50)

    def box_iou(self, box1, box2):
        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
                 torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        # iou = inter / (area1 + area2 - inter)
        return inter / (area1[:, None] + area2 - inter)



def ap_per_class(tp, conf, pred_cls, target_cls):
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, ap, f1, unique_classes.astype('int32')

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec
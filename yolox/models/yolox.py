#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import numpy as np
import cv2
import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        #debug
        #self.show_train(x, targets)

        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, lmk_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "lmk_loss": lmk_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

    def show_train(self, x, targets):
        print("show_train .......   ")
        rgb_means = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        x = x.cpu().numpy()
        if targets is not None:
            targets = targets.cpu().numpy()
            x[..., 0] = x[..., 0] * std[0] + rgb_means[0]
            x[..., 1] = x[..., 1] * std[1] + rgb_means[1]
            x[..., 2] = x[..., 2] * std[2] + rgb_means[2]
            x = np.clip(x, 0, 1.0)
            x = x * 255.0
            x = np.array(x, dtype=np.uint8)

            for i, target in enumerate(targets[0:x.shape[0]]):
                img = x[i]
                img = np.transpose(img, (1,2,0))
                img = img[:,:,::-1]
                img_cv = cv2.resize(img, (img.shape[1], img.shape[0]))
                boxes = target[...,1:5]
                cls = target[...,0]
                lmks = target[...,-10:]
                for j, box in enumerate(boxes):
                    xj, yj, w, h = np.array(boxes[j], dtype=np.int32)
                    x0 = int(xj - 0.5 * w)
                    y0 = int(yj - 0.5 * h)
                    x1 = int(xj + 0.5 * w)
                    y1 = int(yj + 0.5 * h)
                    if w == 0 or h == 0:
                        continue
                    #cv2.rectangle(img_cv,(x0, y0 + 1),(x0 + 1, y0),(0,0,255))
                    img_cv = cv2.rectangle(img_cv, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    #cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
                    # landmarks
                    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
                    for k in range(5):
                        point_x = int(lmks[j][2 * k])
                        point_y = int(lmks[j][2 * k + 1])
                        if point_x > 0 and point_y > 0:
                            cv2.circle(img_cv, (point_x, point_y), 3, clors[k], -1)
                cv2.imwrite("./1_" + str(i)+'_'+str(j)+'.jpg', img_cv)
            exit(0)


# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from yolox.data import YOLODataset, ValTransform
from yolox.data import YOLO_CLASSES
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.output_dir = "./output/YOLOX_outputs"
        self.print_interval = 10
        self.num_classes = len(YOLO_CLASSES)
        self.depth = 0.33
        self.width = 0.375   # min - 0.375 ; s - 0.50
        self.scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.eval_interval = 10
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            YOLODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = YOLODataset(
            data_dir='/home/xialuxi/work/dukto/data/yolov5face/yoloface/train_data',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = InfiniteSampler(
                len(self.dataset), seed=self.seed if self.seed else 0
            )
        else:
            #sampler = torch.utils.data.RandomSampler(self.dataset)
            sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )


        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):

        valdataset = YOLODataset(
            data_dir='/home/xialuxi/work/dukto/data/yolov5face/yoloface/train_data',
            train=False,
            img_size=self.test_size,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        #dataloader_kwargs["collate_fn"] =valdataset.collate_fn
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import VOCEvaluator, YOLOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = YOLOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator

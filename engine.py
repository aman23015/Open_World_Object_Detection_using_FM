# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# -----------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import os
import torch
# import util.misc as utils
# from datasets.open_world_eval import OWEvaluator
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

import os
import shutil
import functools
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import torch
from utils import *
from collections import defaultdict


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def parse_rec(filename, known_classes):
    """ Parse a PASCAL VOC xml file """

    VOC_CLASS_NAMES_COCOFIED = [
        "airplane", "dining table", "motorcycle",
        "potted plant", "couch", "tv"
    ]
    BASE_VOC_CLASS_NAMES = [
        "aeroplane", "diningtable", "motorbike",
        "pottedplant", "sofa", "tvmonitor"
    ]

    tree = ET.parse(filename)
    # import pdb;pdb.set_trace()
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        cls_name = obj.find('name').text
        if cls_name in BASE_VOC_CLASS_NAMES:
            cls_name = VOC_CLASS_NAMES_COCOFIED[BASE_VOC_CLASS_NAMES.index(cls_name)]
        if cls_name not in known_classes:
            cls_name = 'unknown'
        obj_struct['name'] = cls_name

        obj_struct['difficult'] = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=False,
             known_classes=None):
    # --------------------------------------------------------
    # Fast/er R-CNN
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Bharath Hariharan
    # --------------------------------------------------------

    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """

    def iou(BBGT, bb):
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        return ovmax, jmax

    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # read list of images
    
    if isinstance(imagesetfile, list):
        lines = imagesetfile
    else:
        with open(imagesetfile, 'r') as f:
            lines = f.read().splitlines()

    imagenames = [f'{i}-^-{os.path.splitext(x)[0]}' for i, x in enumerate(lines)]

    # load annots
    recs = {}
    if isinstance(annopath, list):
        for a in annopath:
            imagename = os.path.splitext(os.path.basename(a))[0]
            recs[imagename] = parse_rec(a, tuple(known_classes))
    else:
        for imagename in imagenames:
            recs[imagename] = parse_rec(annopath.format(imagename.split('-^-')[-1]), tuple(known_classes))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename.split('-^-')[-1]] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox, 'difficult': difficult,'det': det}
        
    print(npos)
    # read dets
    if isinstance(detpath, list):
        lines = detpath
    else:
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    if len(splitlines) == 0:
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)
    else:
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])


    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        ##todo: class_recs is a dict with image_ides,
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            ovmax, jmax = iou(BBGT, bb)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                    ##todo: here we remove detections with the same annotation
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    '''
    Computing Absolute Open-Set Error (A-OSE) and Wilderness Impact (WI)
                                    ===========    
    Absolute OSE = # of unknown objects classified as known objects of class 'classname'
    WI = FP_openset / (TP_closed_set + FP_closed_set)
    '''
    # logger = logging.getLogger(__name__)

    # Finding GT of unknown objects
    unknown_class_recs = {}
    n_unk = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename.split('-^-')[-1]] if obj["name"] == 'unknown']
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(bool)
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult)
        unknown_class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    if classname == 'unknown':
        return rec, prec, ap, 0., n_unk, None, None

    # Go down each detection and see if it has an overlap with an unknown object.
    # If so, it is an unknown object that was classified as known.
    is_unk = np.zeros(nd)
    for d in range(nd):
        R = unknown_class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            is_unk[d] = 1.0

    is_unk_sum = np.sum(is_unk)
    tp_plus_fp_closed_set = tp+fp
    fp_open_set = np.cumsum(is_unk)

    return rec, prec, ap, is_unk_sum, n_unk, tp_plus_fp_closed_set, fp_open_set

class OWEvaluator(object):
    def __init__(self, voc_gt, args=None, use_07_metric=True, ovthresh=list(range(50, 100, 5))):
        self.use_07_metric = use_07_metric
        self.ovthresh = ovthresh
        self.voc_gt = voc_gt
        self.eps = torch.finfo(torch.float64).eps
        self.num_classes = len(self.voc_gt.CLASS_NAMES)
        self._class_names = self.voc_gt.CLASS_NAMES
        self.AP = torch.zeros(self.num_classes, 1)
        self.all_recs = defaultdict(list)
        self.all_precs = defaultdict(list)
        self.recs = defaultdict(list)
        self.precs = defaultdict(list)
        self.num_unks = defaultdict(list)
        self.unk_det_as_knowns = defaultdict(list)
        self.tp_plus_fp_cs = defaultdict(list)
        self.fp_os = defaultdict(list)
        self.coco_eval = dict(bbox=lambda: None)
        self.coco_eval['bbox'].stats = torch.tensor([])
        self.coco_eval['bbox'].eval = dict()

        self.img_ids = []
        self.lines = []
        self.lines_cls = []

        self.prev_intro_cls = args.PREV_INTRODUCED_CLS
        self.curr_intro_cls = args.CUR_INTRODUCED_CLS
        self.total_num_class = len(voc_gt.CLASS_NAMES)
        self.unknown_class_index = len(voc_gt.KNOWN_CLASS_NAMES)
        self.prev_known_classnames = voc_gt.PREV_KNOWN_CLASS_NAMES


        self.num_seen_classes = self.prev_intro_cls + self.curr_intro_cls
        self.known_classes = self._class_names[:self.num_seen_classes]
        print("testing data details")
        print(self.total_num_class)
        print(self.unknown_class_index)



    def update(self, predictions):
        for img_id, pred in predictions.items():
            pred_boxes, pred_labels, pred_scores = [pred[k].cpu() for k in ['boxes', 'labels', 'scores']]
            self.img_ids.append(img_id)
            classes = pred_labels.tolist()
            for (xmin, ymin, xmax, ymax), cls, score in zip(pred_boxes.tolist(), classes, pred_scores.tolist()):
                xmin += 1
                ymin += 1
                self.lines.append(f"{img_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}")
                self.lines_cls.append(cls)

    def compute_avg_precision_at_many_recall_level_for_unk(self, precisions, recalls):
        precs = {}
        for r in range(1, 10):
            r = r/10
            p = self.compute_avg_precision_at_a_recall_level_for_unk(precisions, recalls, recall_level=r)
            precs[r] = p
        return precs

    def compute_avg_precision_at_a_recall_level_for_unk(self, precisions, recalls, recall_level=0.5):
        precs = {}
        for iou, recall in recalls.items():
            prec = []
            for cls_id, rec in enumerate(recall):
                if cls_id == self.unknown_class_index and len(rec)>0:
                    p = precisions[iou][cls_id][min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))]
                    prec.append(p)
            if len(prec) > 0:
                precs[iou] = np.mean(prec)
            else:
                precs[iou] = 0
        return precs

    def compute_WI_at_many_recall_level(self, recalls, tp_plus_fp_cs, fp_os):
        wi_at_recall = {}
        for r in range(1, 10):
            r = r/10
            wi = self.compute_WI_at_a_recall_level(recalls, tp_plus_fp_cs, fp_os, recall_level=r)
            wi_at_recall[r] = wi
        return wi_at_recall

    def compute_WI_at_a_recall_level(self, recalls, tp_plus_fp_cs, fp_os, recall_level=0.5):
        wi_at_iou = {}
        for iou, recall in recalls.items():
            tp_plus_fps = []
            fps = []
            for cls_id, rec in enumerate(recall):
                if cls_id in range(self.num_seen_classes) and len(rec) > 0:
                    index = min(range(len(rec)), key=lambda i: abs(rec[i] - recall_level))
                    #import ipdb; ipdb.set_trace()
                    tp_plus_fp = tp_plus_fp_cs[iou][cls_id][index]
                    tp_plus_fps.append(tp_plus_fp)
                    fp = fp_os[iou][cls_id][index]
                    fps.append(fp)
            if len(tp_plus_fps) > 0:
                wi_at_iou[iou] = np.mean(fps) / np.mean(tp_plus_fps)
            else:
                wi_at_iou[iou] = 0
        return wi_at_iou

    def synchronize_between_processes(self):
        #self.img_ids = torch.tensor(self.img_ids, dtype=torch.int64)
        self.lines_cls = torch.tensor(self.lines_cls, dtype=torch.int64)
        self.img_ids, self.lines, self.lines_cls = self.merge(self.img_ids, self.lines, self.lines_cls)

    def merge(self, img_ids, lines, lines_cls):
        flatten = lambda ls: [s for l in ls for s in l]

        all_img_ids = flatten(all_gather(img_ids))
        all_lines_cls = torch.cat(all_gather(lines_cls))
        all_lines = flatten(all_gather(lines))
        return all_img_ids, all_lines, all_lines_cls

    def accumulate(self):
        for class_label_ind, class_label in enumerate(self._class_names):
            lines_by_class = [l + '\n' for l, c in zip(self.lines, self.lines_cls.tolist()) if c == class_label_ind]
            if len(lines_by_class) == 0:
                lines_by_class = []
            print(class_label + " has " + str(len(lines_by_class)) + " predictions.")
            ovthresh = 50
            ovthresh_ind, _ = map(self.ovthresh.index, [50, 75])
            
            self.rec, self.prec, self.AP[class_label_ind, ovthresh_ind], self.unk_det_as_known, \
                self.num_unk, self.tp_plus_fp_closed_set, self.fp_open_set = voc_eval(lines_by_class, \
                self.voc_gt.annotations, self.voc_gt.image_set, class_label, ovthresh=ovthresh / 100.0, use_07_metric=self.use_07_metric, known_classes=self.known_classes) #[-1]

            self.AP[class_label_ind, ovthresh_ind] = self.AP[class_label_ind, ovthresh_ind] * 100
            self.all_recs[ovthresh].append(self.rec)
            self.all_precs[ovthresh].append(self.prec)
            self.num_unks[ovthresh].append(self.num_unk)
            self.unk_det_as_knowns[ovthresh].append(self.unk_det_as_known)
            self.tp_plus_fp_cs[ovthresh].append(self.tp_plus_fp_closed_set)
            self.fp_os[ovthresh].append(self.fp_open_set)
            try:
                self.recs[ovthresh].append(self.rec[-1] * 100)
                self.precs[ovthresh].append(self.prec[-1] * 100)
            except:
                self.recs[ovthresh].append(0.)
                self.precs[ovthresh].append(0.)

    def summarize(self, fmt='{:.06f}'):
        o50, _ = map(self.ovthresh.index, [50, 75])
        mAP = float(self.AP.mean())
        mAP50 = float(self.AP[:, o50].mean())
        print('detection mAP50:', fmt.format(mAP50))
        print('detection mAP:', fmt.format(mAP))
        print('---AP50---')
        wi = self.compute_WI_at_many_recall_level(self.all_recs, self.tp_plus_fp_cs, self.fp_os)
        print('Wilderness Impact: ' + str(wi))
        avg_precision_unk = self.compute_avg_precision_at_many_recall_level_for_unk(self.all_precs, self.all_recs)
        print('avg_precision: ' + str(avg_precision_unk))
        total_num_unk_det_as_known = {iou: np.sum(x) for iou, x in self.unk_det_as_knowns.items()} #torch.sum(self.unk_det_as_knowns[:, o50]) #[np.sum(x) for x in self.unk_det_as_knowns[:, o50]]
        total_num_unk = self.num_unks[50][0]
        print('Absolute OSE (total_num_unk_det_as_known): ' + str(total_num_unk_det_as_known))
        print('total_num_unk ' + str(total_num_unk))
        print("AP50: " + str(['%.1f' % x for x in self.AP[:, o50]]))
        print("Precisions50: " + str(['%.1f' % x for x in self.precs[50]]))
        print("Recall50: " + str(['%.1f' % x for x in self.recs[50]]))

        if self.prev_intro_cls > 0:
            print("Prev class AP50: " + str(self.AP[:, o50][:self.prev_intro_cls].mean()))
            print("Prev class Precisions50: " + str(np.mean(self.precs[50][:self.prev_intro_cls])))
            print("Prev class Recall50: " + str(np.mean(self.recs[50][:self.prev_intro_cls])))

        print("Current class AP50: " + str(self.AP[:, o50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls].mean()))
        print("Current class Precisions50: " + str(np.mean(self.precs[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls])))
        print("Current class Recall50: " + str(np.mean(self.recs[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls])))

        print("Known AP50: " + str(self.AP[:, o50][:self.prev_intro_cls + self.curr_intro_cls].mean()))
        print("Known Precisions50: " + str(np.mean(self.precs[50][:self.prev_intro_cls + self.curr_intro_cls])))
        print("Known Recall50: " + str(np.mean(self.recs[50][:self.prev_intro_cls + self.curr_intro_cls])))

        print("Unknown AP50: " + str(self.AP[:, o50][-1]))
        print("Unknown Precisions50: " + str(self.precs[50][-1]))
        print("Unknown Recall50: " + str(self.recs[50][-1]))

        for class_name, ap in zip(self._class_names, self.AP[:, o50].cpu().tolist()):
            print(class_name, fmt.format(ap))
        self.coco_eval['bbox'].stats = torch.cat(
            [self.AP[:, o50].mean(dim=0, keepdim=True),
             self.AP.flatten().mean(dim=0, keepdim=True), self.AP.flatten()])
        
        Res  = {
            "WI":wi[0.8][50],
            "AOSA": total_num_unk_det_as_known[50],
            
            "CK_AP50": float(self.AP[:, o50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls].mean().detach().cpu()),
            "CK_P50": np.mean(self.precs[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls]),
            "CK_R50": np.mean(self.recs[50][self.prev_intro_cls:self.prev_intro_cls + self.curr_intro_cls]),
            
            "K_AP50": float(self.AP[:, o50][:self.prev_intro_cls + self.curr_intro_cls].mean().detach().cpu()),
            "K_P50": np.mean(self.precs[50][:self.prev_intro_cls + self.curr_intro_cls]),
            "K_R50": np.mean(self.recs[50][:self.prev_intro_cls + self.curr_intro_cls]),
            
            "U_AP50": float(self.AP[:, o50][-1].detach().cpu()),
            "U_P50": self.precs[50][-1],
            "U_R50": self.recs[50][-1]
        }
        if self.prev_intro_cls > 0:
            Res["PK_AP50"] = float(self.AP[:, o50][:self.prev_intro_cls].mean().detach().cpu())
            Res["PK_P50"] = np.mean(self.precs[50][:self.prev_intro_cls])
            Res["PK_R50"] =np.mean(self.recs[50][:self.prev_intro_cls])
        
        return Res


@torch.no_grad()
def evaluate(model, postprocessors, data_loader, base_ds, device, output_dir, args):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    coco_evaluator = OWEvaluator(base_ds, args=args)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples.tensors)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        image_ids = [''.join([chr(int(t)) for t in target['image_id']]) for target in targets]
        if len(set(image_ids)) != len(image_ids):
            import ipdb;
            ipdb.set_trace()
        res = {''.join([chr(int(t)) for t in target['image_id']]): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        res = coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['metrics'] = res
    if coco_evaluator is not None:
        stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator


@torch.no_grad()
def viz(model, postprocessors, data_loader, device, output_dir, base_ds, args):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Viz:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples.tensors)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors(outputs, orig_target_sizes, args.viz)
        plot_batch(samples.tensors, results, targets, args.output_dir,
                   [''.join([chr(int(i)) for i in target['image_id']]) + '.jpg' for target in targets],
                   base_ds.KNOWN_CLASS_NAMES + ['unknown'], orig_target_sizes)

    return


@torch.no_grad()
def plot_batch(samples, results, targets, output_dir, image_names, cls_names, orig_target_sizes):
    for i, r in enumerate(results):
        img = samples[i].swapaxes(0, 1).swapaxes(1, 2).detach().cpu()
        plot_bboxes_on_image({k: v.detach().cpu() for k, v in r.items()}, img.numpy(), output_dir, image_names[i],
                             cls_names, num_known=sum(targets[i]['labels'] < len(cls_names) - 1),
                             num_unknown=sum(targets[i]['labels'] == len(cls_names) - 1), img_size=orig_target_sizes[i])

    return


def plot_bboxes_on_image(detections, img, output_dir, image_name, cls_names, num_known=10, num_unknown=5,
                         img_size=None):
    os.makedirs(output_dir, exist_ok=True)
    img = img * np.array([0.26862954, 0.26130258, 0.27577711]) + np.array([0.48145466, 0.4578275, 0.40821073])
    # Extract detections from dictionary
    # import ipdb; ipdb.set_trace()
    if True:
        unk_ind = detections['labels'] == len(cls_names) - 1
        unk_s = detections['scores'][unk_ind]
        unk_l = detections['labels'][unk_ind]
        unk_b = detections['boxes'][unk_ind]
        unk_s, indices = unk_s.topk(min(num_unknown + 1, len(unk_s)))
        unk_l = unk_l[indices]
        unk_b = unk_b[indices]

        k_s = detections['scores'][~unk_ind]
        k_l = detections['labels'][~unk_ind]
        k_b = detections['boxes'][~unk_ind]
        k_s, indices = k_s.topk(min(num_known + 3, len(k_s)))
        k_l = k_l[indices]
        k_b = k_b[indices]
        scores = torch.cat([k_s, unk_s])
        labels = torch.cat([k_l, unk_l])
        boxes = torch.cat([k_b, unk_b])
    else:
        scores = detections['scores']
        labels = detections['labels']
        boxes = detections['boxes']

    fig, ax = plt.subplots(1)
    plt.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    img_size = (img_size * 840 / img_size.max()).detach().cpu().numpy().astype('int32')
    ax.imshow(img[:img_size[0], :img_size[1], :])

    # Plot bounding boxes on image
    for i in range(len(labels)):
        score = scores[i]
        label = cls_names[int(labels[i])]
        if (label == 'unknown' and score > -0.025) or \
                (label != 'unknown' and score > 0.25) or label == 'fish':

            box = boxes[i]

            xmin, ymin, xmax, ymax = [int(b) for b in box.numpy().astype(np.int32)]
            if label == 'unknown':
                rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='b', facecolor='none')
            else:
                rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='g', facecolor='none')

            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f'{label}: {score:.2f}', fontsize=10, color='g')
    plt.savefig(os.path.join(output_dir, image_name), dpi=300, bbox_inches='tight', pad_inches=0)
    return

import os
import tqdm
import json
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps
from IPython import embed


def norm_score(pred):
    """norm score
    args:
        pred {image_id: [[x1, y1, x2, y2, s],]}
    """
    max_score = 0
    min_score = 1
    for key, value in pred.items():
        if len(value) == 0:
            continue
        _min = np.min(value[:, -1])
        _max = np.max(value[:, -1])
        max_score = max(_max, max_score)
        min_score = min(_min, min_score)

    diff = max_score - min_score
    for key, value in pred.items():
        if len(value) == 0:
            continue
        value[:, -1] = (value[:, -1] - min_score) / diff


# gt
def get_gt_from_json(gt_file):
    """gt_file : json file for image url with image bbox
    Returns:
        gt_dict : {image_id: [[x1, y1, x2, y2]]}
    """
    gt_list = open(gt_file).readlines()
    image_list = []
    bbox_list = []
    for data in gt_list:
        image_url = json.loads(data.strip())["image_url"]
        img_lbl = json.loads(data.strip())["image_bbox"]
        image_bbox = []
        for idx, label in enumerate(img_lbl):
            x1, x2, x3, x4 = label["x1"], label["x2"], label["x3"], label["x4"]
            y1, y2, y3, y4 = label["y1"], label["y2"], label["y3"], label["y4"]
            image_bbox.append([int(x1), int(y1), int(x3), int(y3)])
        image_list.append(image_url)
        bbox_list.append(image_bbox)
    gt_dict = {}
    for image_id, bbox_id in zip(image_list, bbox_list):
        image_key = image_id.split('/')[-1].split('.')[0]
        bbox_array = np.array(bbox_id).astype('float')
        gt_dict[image_key] = bbox_array

    return gt_dict


# preds
def get_preds(pred_folder):
    """
    Args:
        pred_folder : a foder for image pred txt with the image name, face count and face bbox with score
    Returns:
        pred_dict: a dict that the image_id: bbox_score array (n, 5)
    """
    pred_file_list = os.listdir(pred_folder)
    pred_dict = {}
    for pred_file in pred_file_list:
        pred_file_path = os.path.join(pred_folder, pred_file)
        pred_list = open(pred_file_path).readlines()
        pred_bbox_score = []
        for data in pred_list[2:]:
            bbox_score = [float(x) for x in data.split(' ')[:5]]
            # bbox_score[2] = bbox_score[2] + bbox_score[0]
            # bbox_score[3] = bbox_score[3] + bbox_score[1]
            pred_bbox_score.append(bbox_score)
        image_key = pred_file.split('.')[0]
        image_bbox_score = np.array(pred_bbox_score).astype('float')
        pred_dict[str(image_key)] = image_bbox_score
    return pred_dict


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    # print(_pred[:, :4].dtype)
    # print(_gt.dtype)
    overlaps = bbox_overlaps(_pred[:, :4], _gt)
    # print(overlaps)
    # print(_pred.shape)
    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
        # print(recall_list)
        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

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


def evaluation(pred_path, gt_path, iou_thresh=0.5):
    pred = get_preds(pred_path)
    norm_score(pred)
    # facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    gt_dict = get_gt_from_json(gt_path)
    thresh_num = 1000
    count_face = 0
    pr_curve = np.zeros((thresh_num, 2)).astype('float')
    for key, value in gt_dict.items():
        if key in pred.keys():
            # print(1)
            pred_info = pred[str(key)]
            gt_boxes = value.astype('float')
            count_face += len(gt_boxes)
            if len(gt_boxes) == 0 or len(pred_info) == 0:
                continue
            # if len(gt_boxes) > 1:
            #     continue
            # if len(gt_boxes)
            ignore = np.zeros(gt_boxes.shape[0])
            # 全部保留
            keep_index = np.array([i for i in range(len(gt_boxes))])
            # print(keep_index)
            count_face += len(keep_index)
            if len(keep_index) != 0:
                ignore[keep_index] = 1
            # print(pred_info.dtype)
            # print(gt_boxes.dtype)
            # print(ignore)
            pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)
            _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
            pr_curve += _img_pr_info

    pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
    propose = pr_curve[:, 0]
    recall = pr_curve[:, 1]
    # print(recall)
    ap = voc_ap(recall, propose)
    return ap


if __name__ == "__main__":
    # pred_path = "/data/remote/github_code/face_detection/Pytorch_Retinaface/detect_validaiton/widerface_pretrain_yewushuju_version"
    pred_path = "/data/remote/github_code/face_detection/Pytorch_Retinaface/detect_validaiton/widerface_version"
    gt_path = "/data/remote/dataset/face_detection/val_face_detection.log"
    AP = evaluation(pred_path, gt_path)
    print("==================== Results ====================")
    print("AP result {}".format(AP))
    print("=================================================")
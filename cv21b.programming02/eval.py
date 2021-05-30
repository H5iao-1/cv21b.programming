import numpy as np
import json

test_images=json.load(open("cv21b.programming02-dataset/val/val.json",'r'))
category_mapping=[]
for image_name in test_images:
    for object_id in test_images[image_name]['objects']:
        category = test_images[image_name]['objects'][object_id]['category']
        if category not in category_mapping:
            category_mapping.append(category)
category_num=len(category_mapping)
print("category num:",category_num)

def cal_iou(pred, gt):
    i_xmin=max(gt[0],pred[0])
    i_ymin = max(gt[1], pred[1])
    i_xmax = min(gt[2], pred[2])
    i_ymax = min(gt[3], pred[3])
    inter = max(0,(i_xmax-i_xmin)*(i_ymax-i_ymin))
    sum_area = max(0,gt[2]-gt[0])*max(0,gt[3]-gt[1])+max(0,pred[2]-pred[0])*max(0,pred[3]-pred[1])-inter
    if sum_area!=0:
        return inter / sum_area
    else:
        return 0

def cal_cls_iou_matrix(gt_objects, pred_objects):
    gt_cls = np.array([category_mapping.index(i['category'])+1 for i in gt_objects])
    pred_cls = np.array([category_mapping.index(i['category'])+1 for i in pred_objects])

    res = [cal_iou(i['bbox'], j['bbox']) for i in pred_objects for j in gt_objects]
    iou_matrix = np.array(res).reshape((len(pred_cls), len(gt_cls)))

    return gt_cls, pred_cls, iou_matrix

def update_ious(gt_image, pred_image, gt_ious, pred_ious, pred_ious_category):
    gt_cls, pred_cls, iou_matrix = cal_cls_iou_matrix([gt_image['objects'][object_id] for object_id in gt_image['objects']],
                                                      [pred_image['objects'][object_id] for object_id in pred_image['objects']])

    gt_match = np.expand_dims(gt_cls, 0).repeat(len(pred_cls), axis=0)
    pred_match = np.expand_dims(pred_cls, 1).repeat(len(gt_cls), axis=1)

    # pred, gt
    iou_matrix[gt_match != pred_match] = -1
    gt_iou_matrix = iou_matrix.copy()
    _gt_ious = np.zeros(len(gt_cls))
    pred_iou_matrix = iou_matrix.copy()
    _pred_ious = np.zeros(len(pred_cls))

    _pred_ious_category = []
    _pred_ious_category_count = []
    for c in range(category_num):
        _pred_ious_category.append(np.zeros(len(np.nonzero(pred_cls == (c+1))[0])))
        _pred_ious_category_count.append(0)
        # if len(np.nonzero(pred_cls == (c+1))[0])!=0:
        #     print(c,len(np.nonzero(pred_cls == (c+1))[0]))

    for j in range(min(len(pred_cls), len(gt_cls))):
        gt_max_ious = gt_iou_matrix.max(axis=0)  # each gt choose max pred
        gt_argmax_ious = gt_iou_matrix.argmax(axis=0)
        gt_iou = gt_max_ious.max(axis=0)  # best covered gt
        gt_ind = gt_max_ious.argmax(axis=0)
        if gt_iou >= 0:
            gt_pred_ind = gt_argmax_ious[gt_ind]  # pred for best covered gt
            _gt_ious[j] = gt_iou_matrix[gt_pred_ind, gt_ind]  # iou for best match pred&gt
            assert _gt_ious[j] == gt_iou
            gt_iou_matrix[gt_pred_ind, :] = -1
            gt_iou_matrix[:, gt_ind] = -1

        pred_max_ious = pred_iou_matrix.max(axis=1)  # each pred choose max gt
        pred_argmax_ious = pred_iou_matrix.argmax(axis=1)
        pred_iou = pred_max_ious.max(axis=0)  # best covered pred
        pred_ind = pred_max_ious.argmax(axis=0)
        if pred_iou >= 0:
            pred_gt_ind = pred_argmax_ious[pred_ind]  # pred for best covered gt
            _pred_ious[j] = pred_iou_matrix[pred_ind, pred_gt_ind]  # iou for best match pred&gt
            assert _pred_ious[j] == pred_iou
            pred_iou_matrix[pred_ind, :] = -1
            pred_iou_matrix[:, pred_gt_ind] = -1

            c=pred_cls[pred_ind]-1
            # print(c)
            # print(_pred_ious_category[c])
            # print(_pred_ious_category_count[c])
            _pred_ious_category[c][_pred_ious_category_count[c]] = pred_iou
            _pred_ious_category_count[c]=_pred_ious_category_count[c]+1

    gt_ious.append(_gt_ious)
    pred_ious.append(_pred_ious)
    for c in range(category_num):
        pred_ious_category[c].append(_pred_ious_category[c])

if __name__ == '__main__':
    # TODO change "my_val.json" to your prediction file
    pred_images = json.load(open('my_val.json'))
    gt_images = json.load(open('cv21b.programming02-dataset/val/val.json'))
    gt_ious, pred_ious, pred_ious_category = [], [], [[] for c in range(category_num)]
    for image_id in gt_images:
        gt_image, pred_image = gt_images[image_id], pred_images[image_id]
        update_ious(gt_image, pred_image, gt_ious, pred_ious, pred_ious_category)

    gt_ious = np.concatenate(gt_ious)
    pred_ious = np.concatenate(pred_ious)
    for c in range(category_num):
        pred_ious_category[c] = np.concatenate(pred_ious_category[c])
    # print(gt_ious.shape)
    # print(pred_ious.shape)
    step = 0.05
    thresholds = np.arange(0.5, 0.95 + 1e-5, step, dtype=np.float32)
    recalls = np.zeros_like(thresholds)
    precisions = np.zeros_like(thresholds)
    precisions_category = np.zeros((category_num, thresholds.shape[0]))
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_ious >= t).sum() / float(gt_ious.shape[0])
        precisions[i] = (pred_ious >= t).sum() / float(pred_ious.shape[0])
        for c in range(category_num):
            if pred_ious_category[c].shape[0]==0:
                precisions_category[c][i]=0
            else:
                precisions_category[c][i] = (pred_ious_category[c] >= t).sum() / float(pred_ious_category[c].shape[0])
    ar = recalls.mean()
    ap = precisions.mean()
    ap_category = precisions_category.mean(axis=1).mean(axis=0)
    print(f"mAP: {ap_category:.2f}, AP: {ap:.2f}, AR: {ar:.2f}")
    print()

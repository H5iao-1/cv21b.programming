def load_res(gt_path):
    with open(gt_path) as f:
        lines = f.readlines()
        items = [line.strip().split() for line in lines if len(line.strip()) > 0]

    res = {item[0]: item[1] for item in items}
    return res


def cal_acc(gt, pr):
    hit_cnt = 0
    for img_id in gt:
        label = gt[img_id]
        if img_id in pr and pr[img_id] == label:
            hit_cnt += 1

    return hit_cnt * 1.0 / len(gt)


gt_path = 'val_label.txt'
# pr_path = 'val_label.txt'
pr_path = 'C:\\Users\\w2000\\Desktop\\keras-face-recognition\\val.txt'
gt = load_res(gt_path)
pr = load_res(pr_path)
print('ACC: %.4f' % cal_acc(gt, pr))
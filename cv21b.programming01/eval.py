def load_res(res_path):
    with open(res_path, 'r') as f:
        items = f.readlines()
        items = [item.strip().split() for item in items]
    iid_to_cid = {item[0]: item[1] for item in items}
    return iid_to_cid


def cal_acc(anno, pred):
    sample_num = len(anno)
    hit_cnt = 0.0
    for iid, cid in anno.items():
        if iid in pred and cid == pred[iid]:
            hit_cnt += 1
    return hit_cnt / sample_num


anno_path = 'val_anno.txt'
pred_path = ''
anno = load_res(anno_path)
pred = load_res(pred_path)
acc = cal_acc(anno, pred)
print('accuracy: %.4f' % acc)




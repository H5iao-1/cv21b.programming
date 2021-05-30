import os
from PIL import Image
import numpy as np
from tqdm import tqdm


class Dataset_Seg():
    def __init__(self, imgs_dir, anno_dir=None, split='test'):
        self.imgs_dir = imgs_dir
        self.anno_dir = anno_dir
        with open('{}.txt'.format(split), 'r') as f:
            self.id_list = f.readlines()

    def load_img(self, fpath):
        return np.asarray(Image.open(fpath), dtype=np.int32)

    def get_loader(self):
        preds = []
        gts = []
        for id in self.id_list:
            id = id.replace('\r', '').replace('\n', '').replace(' ', '')
            preds.append(self.load_img(os.path.join(self.imgs_dir, id + '.png')))
            gts.append(self.load_img(os.path.join(self.anno_dir, id + '.png')))
        return zip(preds, gts)


def cal_mIOU(confusion_matrix):
    miou = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
    return np.nanmean(miou)


def cal_pixelAcc(confusion_matrix):
    return np.diag(confusion_matrix).sum() / confusion_matrix.sum()


def cal_mAcc(confusion_matrix):
    cAcc = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1))
    mAcc = np.nanmean(cAcc)
    return mAcc


def evaluate(loader, num_class):
    c_mtxs = []
    for (pred, gt) in tqdm(loader):
        mask = (gt >= 0) & (gt < num_class)
        label = num_class * gt[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=num_class ** 2)
        confusion_matrix = count.reshape(num_class, num_class)
        c_mtxs.append(confusion_matrix)
    c_mtxs = np.sum(np.array(c_mtxs), axis=0)
    pacc = cal_pixelAcc(c_mtxs)
    macc = cal_mAcc(c_mtxs)
    miou = cal_mIOU(c_mtxs)

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}'.format(miou))
    print('Pixel Accuracy: {:.4f}, Mean Accuracy: {:.4f}'.format(pacc, macc))


if __name__ == '__main__':
    # You can change 'split' param from 'test' to 'val',
	# and if your 'eval.py' and 'val.txt' are not in the same dir, simply change it to an absolute path,
	# e.g., Dataset_Seg(r'path to your prediction results', r'path to the groundtruths', split='/data/workspace/val') (assume you have a '/data/workspace/val.txt')
    loader = Dataset_Seg(r'', r'', split='test').get_loader()
    evaluate(loader, 21)

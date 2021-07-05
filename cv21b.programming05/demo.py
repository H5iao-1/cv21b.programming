import string
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import os
from utils import CTCLabelConverter, AttnLabelConverter
#from dataset import RawDataset, AlignCollate,MyDataSet
from dataset import*
from model import Model
from tqdm import tqdm
import json
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    # checkpoint = torch.load(opt.saved_model, map_location='cpu')
    # model_dict = model.state_dict()
    # for k,v in model_dict.items():
    #    if  (k != 'module.Transformation.GridGenerator.P_hat' and k != 'module.Prediction.attention_cell.rnn.weight_ih' and k!='module.Prediction.generator.weight' and k!='module.Prediction.generator.bias'):
    #     model_dict[k] = checkpoint[k]
    # model.load_state_dict(model_dict)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    dic = {}
    with torch.no_grad():
        real=0;
        for image_tensors, image_path_list in tqdm(demo_loader):
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                start = time.time()
                preds = model(image, text_for_pred, is_train=False)
                end = time.time()
                print("循环运行时间:%.2f秒"%(end-start))

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            # log = open(f'./log_demo_result.txt', 'a')
            # dashed_line = '-' * 80
            # head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            #
            # print(f'{dashed_line}\n{head}\n{dashed_line}')
            # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # # calculate confidence score (= multiply of pred_max_prob)
                # confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                # # if confidence_score<0.8:
                # print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                # log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
                #
                # gt=img_name.split('_')[2].replace(".jpg","")
                # if gt==pred:
                #     real=real+1
                # else:
                #     print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                #     log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
                name = img_name.split("/")[1]
                dic[name] = pred
                # print(pred)
        with open("test.json", "w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False)

        print(len(dic))



if __name__ == '__main__':
    # python demo.py  --saved_model saved_models/TPS-RCNN-BiLSTM-CTC-Seed1111/best_accuracy.pth
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default="dataset/test/", help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--saved_model', default="saved_models/TPS-RCNN-BiLSTM-CTC-Seed1111/best_accuracy.pth", help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')

    parser.add_argument('--character', type=str,
                        default='邓玲强英飞郑芳宏田凤正红梅学杰德平玉敏荣瑞云武亚娟沈良韩少海宋生林祥陈曹方秀永晓慧俊霞朱志宇珍 春赵孙龙冯叶艳家兴安华国吴新周徐峰杨光罗梁大雪江高文唐张兰建振佳东利黄勇燕辉军庆何宝马福金元丽胡明伟谢天程斌刘洪成君郭鹏王清李忠蔡萍许子立世波美',
                        help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default="TPS", help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default="RCNN", help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default="CTC", help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    print(type(opt))

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6] # same with ASTER setting (use 94 char).

    # with open('label.txt','r') as f:
    #     opt.character=f.read()

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)

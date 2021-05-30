import os
import time
import json
from tqdm import tqdm

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from faster_rcnn_framework import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from resnet50_fpn_model import resnet50_fpn_backbone


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # models = {}
    # img_list = list(os.listdir("dataset/val/val"))
    #
    # a = set()
    # for img_name in tqdm(img_list):
    #
    #     # load image
    #     original_img = Image.open("dataset/val/val/" + img_name)
    #
    #     # from pil image to tensor, do not normalize image
    #     data_transform = transforms.Compose([transforms.ToTensor()])
    #     img = data_transform(original_img)
    #     # expand batch dimension
    #     img = torch.unsqueeze(img, dim=0)
    #     height, width = img.shape[-2:]
    #     a.add((height, width))
    # for i in list(a):
    #     models[i] =
    #
    # return

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    thresh = 0.3
    model = create_model(num_classes=472)

    # load train weights
    train_weights = "./save_weights/resNetFpn-model-0.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)
    model.eval()  # 进入验证模式

    # read class_indict
    label_json_path = './dataset_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    img_list = list(os.listdir("dataset/test/test"))
    img_dict = {}
    count = 0
    for img_name in tqdm(img_list):
        original_img = Image.open("dataset/test/test/" + img_name)
        h, w = original_img.height, original_img.width
        if (h, w) in img_dict:
            img_dict[(h, w)].append(img_name)
        else:
            img_dict[(h, w)] = [img_name]
    result = {}
    # count = 0
    for hw in tqdm(img_dict):
        with torch.no_grad():
            init_img = torch.zeros((1, 3, hw[0], hw[1]), device=device)
            model(init_img)
        for img_name in img_dict[hw]:

            ano = {}
            objects = {}
            # load image
            original_img = Image.open("dataset/val/val/" + img_name)

            # from pil image to tensor, do not normalize image
            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                # init
                # img_height, img_width = img.shape[-2:]
                # init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                # model(init_img)

                predictions = model(img.to(device))[0]

                predict_boxes = predictions["boxes"].to("cpu").numpy()
                predict_classes = predictions["labels"].to("cpu").numpy()
                predict_scores = predictions["scores"].to("cpu").numpy()

                if len(predict_boxes) == 0:
                    print("没有检测到任何目标!")

                ano["height"] = int(original_img.height)
                ano["width"] = int(original_img.width)
                ano["depth"] = 3

                for i in range(predict_boxes.shape[0]):
                    if predict_scores[i] > thresh:
                        box = list(predict_boxes[i].tolist())
                        for k in range(len(box)):
                            box[k] = int(box[k])
                        o = {}
                        o["category"] = str(category_index[predict_classes[i]])
                        o["bbox"] = box
                        objects[str(i)] = o
                        # objects[category_index[predict_classes[i]]] =
                # print(original_img.height)
                # print(original_img.width)
                ano["objects"] = objects
                result[img_name] = ano
                # count += 1
                # if count % 50 == 0:
                #     print(count)
    f = open('my_test.json', 'a')
    json.dump(result, f, indent=1)  # 写入文件后，单引号会被转换成双引号
    f.close()


if __name__ == '__main__':
    main()



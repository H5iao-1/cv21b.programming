from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class MyDataSet(Dataset):

    def __init__(self, dataset_root, transforms, name):
        self.root = os.path.join(dataset_root, name)
        self.img_root = os.path.join(self.root, name)
        self.info_root = os.path.join(self.root, name + ".json")

        self.info = json.load(open(self.info_root))
        self.img = list(self.info.keys())
        self.annotations = list(self.info.values())

        # read class_indict
        json_file = open('dataset_classes.json')
        self.class_dict = json.load(json_file)

        self.transforms = transforms

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        image_root = os.path.join(self.img_root, self.img[idx])
        image = Image.open(image_root)
        # read xml
        # xml_path = self.xml_list[idx]
        # with open(xml_path) as fid:
        #     xml_str = fid.read()
        # xml = etree.fromstring(xml_str)
        # data = self.parse_xml_to_dict(xml)["annotation"]
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        ano = self.annotations[idx]

        for obj in ano["objects"]:
            xmin = float(ano["objects"][obj]["bbox"][0])
            ymin = float(ano["objects"][obj]["bbox"][1])
            xmax = float(ano["objects"][obj]["bbox"][2])
            ymax = float(ano["objects"][obj]["bbox"][3])

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: " + self.img[idx] + " - there are some bbox w/h <=0")
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[ano["objects"][obj]["category"]])
            iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        ano = self.annotations[idx]
        return int(ano["height"]), int(ano["width"])

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """

        ano = self.annotations[idx]
        data_height, data_width = int(ano["height"]), int(ano["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in ano["objects"]:
            xmin = float(ano["objects"][obj]["bbox"][0])
            ymin = float(ano["objects"][obj]["bbox"][1])
            xmax = float(ano["objects"][obj]["bbox"][2])
            ymax = float(ano["objects"][obj]["bbox"][3])

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[ano["objects"][obj]["category"]])
            iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# import transforms
# from draw_box_utils import draw_box
# from PIL import Image
# import json
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# import random
#
# # read class_indict
# category_index = {}
# try:
#     json_file = open('./dataset_classes.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {v: k for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# # load train data set
# train_data_set = MyDataSet("dataset", data_transform["train"], "train")
# print(len(train_data_set))
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     draw_box(img,
#              target["boxes"].numpy(),
#              target["labels"].numpy(),
#              [1 for i in range(len(target["labels"].numpy()))],
#              category_index,
#              thresh=0.5,
#              line_thickness=5)
#     plt.imshow(img)
#     plt.show()

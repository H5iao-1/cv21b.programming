
import time

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from pspnet_class import Pspnet

if __name__ == "__main__":
    pspnet = Pspnet()

    mode = "predict"

    # 需要预测的
    # "val" or "test"
    file = "val"

    if mode == "predict":
        if file == "test":
            for i in tqdm(range(2100, 2600)):
                img = "dataset/image/" + str(i) + ".jpg"
                try:
                    image = Image.open(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    r_image = pspnet.detect_image(image)
                    r_image.save("result/test/"+str(i)+".png")

        elif file == "val":
            for i in tqdm(range(1600, 2100)):
                img = "dataset/image/" + str(i) + ".jpg"
                try:
                    image = Image.open(img)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    r_image = pspnet.detect_image(image)
                    r_image.save("result/val/"+str(i)+".png")

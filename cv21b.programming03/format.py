import os
from PIL import Image
import numpy as np
from tqdm import tqdm

"""
用于将模型生成的RGB图片转为与原标注图片相同的P模式并保存
"""

if __name__ == '__main__':
    # 原标注图片使用的调色板
    palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]
    # 原标注图片使用的调色板所对应的RGB值
    c = {0: (0, 0, 0), 255: (224, 224, 192), 3: (128, 128, 0), 15: (192, 128, 128), 9: (192, 0, 0), 11: (192, 128, 0), 8: (64, 0, 0), 14: (64, 128, 128), 5: (128, 0, 128), 16: (0, 64, 0), 19: (128, 192, 0), 13: (192, 0, 128), 7: (128, 128, 128), 4: (0, 0, 128), 10: (64, 128, 0), 6: (0, 128, 128), 2: (0, 128, 0), 12: (64, 0, 128), 20: (0, 64, 128), 1: (128, 0, 0), 18: (0, 192, 0), 17: (128, 64, 0)}
    for x in c:
        vector2 = np.array(c[x])
        c[x] = vector2

    # 需要转换的文件
    # "val" or "test"
    file = "val"

    if file == "val":
        for i in tqdm(range(1600, 2100)):
            # 改成需要调整的图片文件路径，修改上面的range
            f1 = "result\\val\\" + str(i) + ".png"
            img = Image.open(f1)
            a = np.asarray(img, dtype=np.int32)
            res = []
            for k in range(0, a.shape[0]):
                res.append([])
                for j in range(0, a.shape[1]):
                    res[k].append(0)
                    vector1 = np.array(a[k][j])
                    m = 10000000
                    for x in c:
                        l = np.linalg.norm(vector1-c[x])
                        if l < m:
                            res[k][j] = x
                            m = l
            res = np.array(res, dtype=np.int32)
            im = Image.fromarray(res)
            converted = Image.new('P', im.size)
            converted.putpalette(palette)
            converted.paste(im, (0, 0))
            # 自行去创建val文件夹
            converted.save("val/"+str(i)+".png")

    elif file == "test":
        for i in tqdm(range(2100, 2600)):
            # 改成需要调整的图片文件路径，修改上面的range
            f1 = "result\\test\\" + str(i) + ".png"
            img = Image.open(f1)
            a = np.asarray(img, dtype=np.int32)
            res = []
            for k in range(0, a.shape[0]):
                res.append([])
                for j in range(0, a.shape[1]):
                    res[k].append(0)
                    vector1 = np.array(a[k][j])
                    m = 10000000
                    for x in c:
                        l = np.linalg.norm(vector1-c[x])
                        if l < m:
                            res[k][j] = x
                            m = l
            res = np.array(res, dtype=np.int32)
            im = Image.fromarray(res)
            converted = Image.new('P', im.size)
            converted.putpalette(palette)
            converted.paste(im, (0, 0))
            # 自行去创建test文件夹
            converted.save("test/"+str(i)+".png")
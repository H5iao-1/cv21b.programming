import shutil
import os

f = open("cv21b.programming01-dataset/val_anno.txt", "r")
file = f.readlines()
os.mkdir("cv21b.programming01-dataset/val")
for i in range(80):
    os.mkdir("cv21b.programming01-dataset/val/"+str(i))
for d in file:
    m = d[0:len(d)-1].split(" ")
    shutil.copyfile("cv21b.programming01-dataset/validation/"+m[0], "cv21b.programming01-dataset/val/"+m[1]+"/"+m[0])


# shutil.copyfile("val/0.jpg", "val1/0/0.jpg")

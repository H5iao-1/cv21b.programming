import json
from tqdm import tqdm


path_train = "dataset/train.json"
path_val = "dataset/val.json"

train_data = json.load((open(path_train, encoding="utf-8")))
val_data = json.load((open(path_val, encoding="utf-8")))

path_prefix = "dataset/train/"
with open("train.txt", "w", encoding="utf-8") as f:
    text = ""
    for element in tqdm(train_data):
        path = path_prefix + element
        name = train_data[element]
        text = text + path + "\n" + name + "\n"
    f.write(text)

path_prefix = "dataset/val/"
with open("val.txt", "w", encoding="utf-8") as f:
    text = ""
    for element in tqdm(val_data):
        path = path_prefix + element
        name = val_data[element]
        text = text + path + "\n" + name + "\n"
    f.write(text)




import json

# 自行修改文件路径
pred_result=json.load(open("val_own.json", encoding='utf-8'))
# pred_result=json.load(open("val.json", encoding='utf-8'))
gt_result=json.load(open("val.json",encoding='utf-8'))

TP = 0
all = 0
for key in gt_result:
    n = len(gt_result[key])
    all += n
    for i in range(n):
        if key in pred_result and len(pred_result[key]) > i:
            if gt_result[key][i] == pred_result[key][i]:
                TP += 1

print("Accuracy", round(TP / all, 3))
cv21b.programming02  物体检测练习

【数据集】
- 471个类别，共39564张图像
- 下载链接：https://pan.baidu.com/s/1pqSYZdsoU_oPzECQW2svXw （提取码：cx8l）
- 训练集
  - 图像数量：23550
  - 图像位置：train/
  - 标注文件：train.json
  - 用途：训练物体检测模型
- 验证集
  - 图像数量：8007
  - 图像位置：val/
  - 标注文件：val.json
  - 用途：使用eval.py进行本地测试
- 测试集
  - 图像数量：8007
  - 图像位置：test/
  - 用途：用于最终测试，因此没有提供标注文件

【评测指标】
- mAP：所有类别上的平均AP
    - AP：在不同Recall下的平均Precision  
参考资料：https://zhuanlan.zhihu.com/p/48693246

【标注文件格式】
{<image_name>:{'height':<height>,'width':<width>,'depth':<depth>,'objects:'{<object_id>:{'category':<category name>,'bbox':\[\<xmin\>,\<ymin\>,\<xmax\>,\<ymax\>\]}}

【任务说明】
1. 使用训练集中的数据训练模型；
2. 使用验证集中的数据调优模型；
3. 采用模型对测试集中的所有图像进行物体检测，提交zip格式，包括：
   - 结果文件命名为“学号.json”，格式同标注文件
   - 汇报幻灯片，命名为“汇报人学号+姓名”
   - 小组构成：小组成员的学号和姓名

【使用流程】

1. 下载数据集并放在 cv21b.programming02-dataset 文件夹下
2. 运行 train_res50_fpn.py 训练模型
3. 运行 predict.py 生成结果文件
4. result 文件夹中包含 PPT，训练了1个epoch的模型权重，结果文件
5. eval.py 用于计算验证集上的mAP。其中验证集上的预测结果需要用训练好的模型&predict.py生成
6. 预训练好的权重和训练好的权重
   https://pan.baidu.com/s/1lYThd30nl8nIXS-MlYqmLg 提取码：8efg

**具体代码解析和其他使用问题，参考 https://b23.tv/HvMiDy**

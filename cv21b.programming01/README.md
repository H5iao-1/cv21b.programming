cv21b.programming01 图像分类练习

【数据集】
- 80个类别，40000张图像
- 训练集
  - 图像数量：20000张图像
  - 图像位置：train/，该目录下包含80个文件夹，文件夹名称即category_id，每个文件夹下保存了该类别的所有图像
  - 用途：训练分类模型
- 验证集
  - 图像数量：10000张图像
  - 图像位置：val/，该目录下保存了验证集的所有图像
  - 标注文件：val_anno.txt，每一行为image_id category_id
  - 用途：使用eval.py进行本地测试
- 测试集
  - 图像数量：10000张图像
  - 图像位置：test/，该目录下保存了测试集的所有图像
  - 用途：用于最终测试，因此没有提供标注文件

【任务说明】

1. 使用训练集中的数据训练模型；
2. 使用验证集中的数据调优模型；
3. 采用模型对测试集中的所有图像进行分类，提交zip格式，包括：
   - 结果文件命名为“学号.txt”，每一行保存一个样本的分类结果，形如：image_id.jpg category_id\n（与val_anno.txt相同）
   - 汇报幻灯片，命名为“汇报人学号+姓名”
   - 小组构成：小组成员的学号和姓名



【使用流程】

1. 解压 cv21b.programming01-dataset.zip
2. 运行 main.py 整理验证集
3. 运行 read_ckpt.py 使用并处理TensorFlow提供的 resnet_v1_50.ckpt 预训练模型
4. 运行 train.py 进行模型训练，将最优的模型存放在 save_weights 中
5. 运行 predict.py 进行测试集预测，并将结果保存在 result.txt 中
6. 其他文件：
   * class_indices.json 用于存放 TensorFlow.Keras 生成的类别字典
   * eval.py 作业一开始提供的，无用
   * model.py ResNet模型搭建
   * Result存放了 Epoch=20 的一次结果，包含save_weight，result.txt，汇报PPT，处理好的预训练模型等。（若想使用，置于正确位置即可）


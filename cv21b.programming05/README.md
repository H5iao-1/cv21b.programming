cv21b.programming05 光学字符识别练习

【数据集】
- 共65014张手写签名图片
- 下载链接：链接：https://pan.baidu.com/s/1Mzfaju7zq1Jk6zVYBp6O-g (提取码：qbow)
- 训练集
  - 图像数量：51950
  - 图像位置：train/
  - 标注文件：train.json
  - 用途：训练OCR模型
- 验证集
  - 图像数量：6540
  - 图像位置：val/
  - 标注文件：val.json
  - 用途：使用eval.py进行本地测试
- 测试集
  - 图像数量：6524
  - 图像位置：test/
  - 用途：用于最终测试，因此没有提供标注文件

【评测指标】
每张图片输出给定数量的预测字符，与真实结果按序匹配
Accuracy=正确预测字符数量/所有真实字符数量

【标注文件格式】
{<image_name>:<text>}

【任务说明】
1. 使用训练集中的数据训练模型；
2. 使用验证集中的数据调优模型；
3. 采用模型对测试集中的所有图像进行物体检测，提交zip格式，包括：
   - 结果文件命名为“学号.json”，格式同标注文件
   - 汇报幻灯片，命名为“汇报人学号+姓名”
   - 小组构成：小组成员的学号和姓名（包括代码下载链接）



【使用流程】

1. 参考项目地址：https://github.com/clovaai/deep-text-recognition-benchmark
2. 相关依赖、运行过程中的文件路径位置相关问题自行解决。
3. 运行 preprocess.py 将 json 文件转为 txt 文件。
4. 运行 create_dataset.py 生成数据库
5. 运行 train.py 进行训练，相关参数说明可间上述项目地址。尤其注意参数 character。
6. 运行 demo.py 进行预测，生成结果。
7. eval.py 为评估文件
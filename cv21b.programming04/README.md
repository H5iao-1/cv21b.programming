cv21b.programming04 人脸识别练习

【任务描述】
将给定的人脸图像与gallery文件夹中的人脸图像进行比对，确认人脸图像身份（即对应gallery中哪张图像）。

【数据集】
- train：训练集，包含100个人（每个人对应一个文件夹，其中有多张人脸图像），用于训练人脸特征表示模型。
注意： 训练集中的人并不在gallery中， 训练集中的数据仅可用于训练如何提取人脸特征（即利用同一个文件夹中的人脸图像属于同一个人、不同文件夹中的人脸图像不属于同一个人）。
- gallery：存档图像，包含50张人脸图像，图像文件名即身份ID。
- val： 验证集，2475张不明身份的人脸图像，用于本地测试（val_label.txt提供了标注）。
评测方法：按照val_label.txt格式生成结果，保存为val_result.txt文件，运行eval.py文件。
- val_label.txt：验证集标注，每行一个图像；第一列对应val目录下的图像名；第二列为身份ID（即所对应的gallery中图像的文件名）。
- test： 测试集，2475张不明身份的人脸图像，用于实际测试（不提供标注）。

【提交内容】
采用模型对测试集中的所有人脸图像进行识别，提交zip格式，包括：
   - 结果文件命名为“学号.txt”，格式与val_label.txt相同
   - 汇报幻灯片，命名为“汇报人学号+姓名”
   - 小组构成：小组成员的学号和姓名（包括代码下载地址）

【使用流程】

1. 下载数据集和模型权重，分别放置在 face_dataset 和 model_data 文件夹下

   链接：https://pan.baidu.com/s/1FtexKz3SRrbwHRwYk35UUA 
   提取码：lvd8

2. 运行 face_recognize.py ，运行过程中的路径相关问题自行解决。

3. 更多使用问题参考
   https://www.bilibili.com/video/BV1wJ411x7tF/?spm_id_from=333.788.recommend_more_video.1
   
4. result - 结果文件


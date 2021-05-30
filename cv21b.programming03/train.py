import os

from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import Adam

from pspnet import pspnet
from pspnet_training import CE, Generator, LossHistory, dice_loss_with_CE
from metrics import Iou_score, f_score

if __name__ == "__main__":
    log_dir = "logs/"

    #   输入图片的大小
    inputs_size = [473, 473, 3]

    #   分类个数+1 (背景
    num_classes = 21

    dice_loss = False

    #   主干网络选择
    #   mobilenet
    backbone = "mobilenet"

    #   是否使用辅助分支
    #   会占用大量显存
    aux_branch = False

    #   下采样的倍数
    downsample_factor = 16

    #   数据集路径
    dataset_path = "dataset/"

    # 获取model
    model = pspnet(num_classes, inputs_size, downsample_factor=downsample_factor, backbone=backbone, aux_branch=aux_branch)

    #   预训练模型
    model_path = "pspnet_mobilenetv2.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # 打开数据集的txt
    with open(os.path.join(dataset_path, "segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()

    # 打开数据集的txt
    with open(os.path.join(dataset_path, "segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir)
    loss_history = LossHistory(log_dir)

    if backbone == "mobilenet":
        freeze_layers = 146
    else:
        freeze_layers = 172

    for i in range(freeze_layers):
        model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    if True:
        lr              = 1e-4
        Init_Epoch      = 0
        Freeze_Epoch    = 50
        Batch_size      = 8

        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                      optimizer = Adam(lr=lr),
                      metrics = [f_score()])

        gen             = Generator(Batch_size, train_lines, inputs_size, num_classes, aux_branch, dataset_path).generate()
        gen_val         = Generator(Batch_size, val_lines, inputs_size, num_classes, aux_branch, dataset_path).generate(False)

        epoch_size      = len(train_lines) // Batch_size
        epoch_size_val  = len(val_lines) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))
        model.fit_generator(gen,
                            steps_per_epoch=epoch_size,
                            validation_data=gen_val,
                            validation_steps=epoch_size_val,
                            epochs=Freeze_Epoch,
                            initial_epoch=Init_Epoch,
                            callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard, loss_history])

    for i in range(freeze_layers):
        model.layers[i].trainable = True

    if True:
        lr              = 1e-5
        Freeze_Epoch    = 50
        Unfreeze_Epoch  = 100
        Batch_size      = 4

        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                      optimizer = Adam(lr=lr),
                      metrics = [f_score()])

        gen             = Generator(Batch_size, train_lines, inputs_size, num_classes, aux_branch, dataset_path).generate()
        gen_val         = Generator(Batch_size, val_lines, inputs_size, num_classes, aux_branch, dataset_path).generate(False)

        epoch_size      = len(train_lines) // Batch_size
        epoch_size_val  = len(val_lines) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))
        model.fit_generator(gen,
                            steps_per_epoch=epoch_size,
                            validation_data=gen_val,
                            validation_steps=epoch_size_val,
                            epochs=Unfreeze_Epoch,
                            initial_epoch=Freeze_Epoch,
                            callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard, loss_history])


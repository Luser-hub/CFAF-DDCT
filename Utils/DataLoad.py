import mne
import numpy as np
import pandas as pd
import scipy.io as scio


def loadData(i,sub_num1):
    datas = []
    labels = []
    evaldata = []
    evallabel = []
    # 读取除了第i个被试外其他所有被试的训练和测试数据和标签
    for j in range(1, sub_num1+1):
        if j == i:
            trainX, testX, label = readRaw(j)
            evaldata.append(trainX)
            evaldata.append(testX)
            evallabel.append(label)
            continue
        trainX, testX, label = readRaw(j)
        datas.append(trainX)
        datas.append(testX)
        labels.append(label)
    datas = np.asarray(datas)
    datas = np.reshape(datas, (-1, 22, 876))
    labels = np.asarray(labels)
    labels = np.reshape(labels, (-1, 4))
    evaldata = np.asarray(evaldata)
    evaldata = np.reshape(evaldata, (-1, 22, 876))
    evallabel = np.asarray(evallabel)
    evallabel = np.reshape(evallabel, (-1, 4))

    return datas[:, :, 0:875]*1000000, labels,evaldata[:, :, 0:875]*1000000,evallabel


"""
数据加载
"""


def shuffleDataset(data, labels, evaldata, evallabel):
    # 处理源域数据（data, labels）
    X_pretrain = data
    Y_pretrain = labels

    # 检查源域标签是否为one-hot格式并转换
    if len(Y_pretrain.shape) > 1 and Y_pretrain.shape[1] > 1:
        Y_pretrain = np.argmax(Y_pretrain, axis=1)
        print(f"已将源域标签从one-hot转换为类别索引，形状: {Y_pretrain.shape}")

    # 处理目标域数据（evaldata, evallabel）
    indices = np.arange(evaldata.shape[0])  # 生成索引
    np.random.shuffle(indices)  # 打乱索引

    shuffled_data = evaldata[indices]  # 打乱数据
    shuffled_label = evallabel[indices]  # 打乱标签

    # 检查目标域标签是否为one-hot格式并转换
    if len(shuffled_label.shape) > 1 and shuffled_label.shape[1] > 1:
        shuffled_label = np.argmax(shuffled_label, axis=1)
        print(f"已将目标域标签从one-hot转换为类别索引，形状: {shuffled_label.shape}")

    # 划分数据集
    total_samples = evaldata.shape[0]
    train_size = int(total_samples * 0.8)  # 80% 训练集

    # 提取训练集和测试集
    X_train = shuffled_data[:train_size]
    Y_train = shuffled_label[:train_size]
    X_test = shuffled_data[train_size:]
    Y_test = shuffled_label[train_size:]

    print(f"数据集划分完成:")
    print(f"  源域数据: {X_pretrain.shape}, 标签: {Y_pretrain.shape}")
    print(f"  目标域训练数据: {X_train.shape}, 标签: {Y_train.shape}")
    print(f"  目标域测试数据: {X_test.shape}, 标签: {Y_test.shape}")

    return X_pretrain, Y_pretrain, X_train, Y_train, X_test, Y_test


def shuffleDataset_Junheng(data, labels, evaldata, evallabel):
    # 处理源域数据（data, labels）
    X_pretrain = data
    Y_pretrain = labels

    # 检查源域标签是否为one-hot格式并转换
    if len(Y_pretrain.shape) > 1 and Y_pretrain.shape[1] > 1:
        Y_pretrain = np.argmax(Y_pretrain, axis=1)
        # print(f"已将源域标签从one-hot转换为类别索引，形状: {Y_pretrain.shape}")

    # 处理目标域数据（evaldata, evallabel）
    indices = np.arange(evaldata.shape[0])  # 生成索引
    np.random.shuffle(indices)  # 打乱索引

    shuffled_data = evaldata[indices]  # 打乱数据
    shuffled_label = evallabel[indices]  # 打乱标签

    # 检查目标域标签是否为one-hot格式并转换
    if len(shuffled_label.shape) > 1 and shuffled_label.shape[1] > 1:
        shuffled_label = np.argmax(shuffled_label, axis=1)
        # print(f"已将目标域标签从one-hot转换为类别索引，形状: {shuffled_label.shape}")

    # 获取所有类别
    classes = np.unique(shuffled_label)

    # 初始化训练集和测试集
    X_train_list = []
    Y_train_list = []
    X_test_list = []
    Y_test_list = []

    # 对每个类别分别进行拆分
    for cls in classes:
        # 获取当前类别的样本索引
        cls_indices = np.where(shuffled_label == cls)[0]

        # 打乱当前类别的样本索引
        np.random.shuffle(cls_indices)

        # 计算当前类别训练集和测试集的大小
        cls_samples = len(cls_indices)
        cls_train_size = int(cls_samples * 0.8)

        # 拆分当前类别的样本
        cls_train_indices = cls_indices[:cls_train_size]
        cls_test_indices = cls_indices[cls_train_size:]

        # 添加到训练集和测试集列表
        X_train_list.append(shuffled_data[cls_train_indices])
        Y_train_list.append(shuffled_label[cls_train_indices])
        X_test_list.append(shuffled_data[cls_test_indices])
        Y_test_list.append(shuffled_label[cls_test_indices])

    # 合并所有类别的训练集和测试集
    X_train = np.vstack(X_train_list) if X_train_list else np.array([])
    Y_train = np.hstack(Y_train_list) if Y_train_list else np.array([])
    X_test = np.vstack(X_test_list) if X_test_list else np.array([])
    Y_test = np.hstack(Y_test_list) if Y_test_list else np.array([])

    # 打乱训练集和测试集
    train_indices = np.arange(len(X_train))
    test_indices = np.arange(len(X_test))
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    X_train = X_train[train_indices]
    Y_train = Y_train[train_indices]
    X_test = X_test[test_indices]
    Y_test = Y_test[test_indices]

    return X_pretrain, Y_pretrain, X_train, Y_train, X_test, Y_test



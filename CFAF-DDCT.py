import time
import mne
from scipy import io
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from Utils.CorrCS import channel_weight_calculation
from Utils.DataLoad import shuffleDataset, shuffleDataset_Junheng
from Utils.SingleLimbDataloader import loadData, loadTargetData, loadSourceData
from Utils.eegconformer import EEGConformer  # 沿用原EEGConformer模型
from Utils.plot_or_print import plot_training_process
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
from braindecode.util import set_random_seeds
from loss_funcs.TransferLoss import TransferLoss
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import torch.nn.functional as F


def preprocess_data_for_loss(source_data, target_data, source_label=None, target_logits=None, source_logits=None,
                             loss_type='mmd'):
    """根据选择的损失函数类型对数据进行预处理"""
    if loss_type in ['mmd', 'spd_mmd']:
        source_batch_size = source_data.size(0)
        target_batch_size = target_data.size(0)
        min_batch_size = min(source_batch_size, target_batch_size)

        if source_batch_size > min_batch_size:
            source_data = source_data[:min_batch_size]
        if target_batch_size > min_batch_size:
            target_data = target_data[:min_batch_size]

        source_flat = source_data.view(min_batch_size, -1)
        target_flat = target_data.view(min_batch_size, -1)

        return source_flat, target_flat

    elif loss_type == 'lmmd':
        if source_label is None or target_logits is None:
            raise ValueError("For LMMD loss, source_label and target_logits are required.")

        source_batch_size = source_data.size(0)
        target_batch_size = target_data.size(0)
        min_batch_size = min(source_batch_size, target_batch_size)

        if source_batch_size > min_batch_size:
            source_data = source_data[:min_batch_size]
            source_label = source_label[:min_batch_size]
        if target_batch_size > min_batch_size:
            target_data = target_data[:min_batch_size]
            target_logits = target_logits[:min_batch_size]

        source_flat = source_data.view(min_batch_size, -1)
        target_flat = target_data.view(min_batch_size, -1)

        return source_flat, target_flat, source_label, target_logits

    elif loss_type == 'coral':
        source_batch_size = source_data.size(0)
        target_batch_size = target_data.size(0)
        min_batch_size = min(source_batch_size, target_batch_size)

        if source_batch_size > min_batch_size:
            source_data = source_data[:min_batch_size]
        if target_batch_size > min_batch_size:
            target_data = target_data[:min_batch_size]

        return source_data, target_data

    elif loss_type in ['adv', 'daan']:
        source_batch_size = source_data.size(0)
        target_batch_size = target_data.size(0)
        min_batch_size = min(source_batch_size, target_batch_size)

        if source_batch_size > min_batch_size:
            source_data = source_data[:min_batch_size]
            if source_logits is not None:
                source_logits = source_logits[:min_batch_size]
        if target_batch_size > min_batch_size:
            target_data = target_data[:min_batch_size]
            if target_logits is not None:
                target_logits = target_logits[:min_batch_size]

        if loss_type == 'daan':
            if source_logits is None or target_logits is None:
                raise ValueError("For DAAN loss, source_logits and target_logits are required.")
            return source_data, target_data, source_logits, target_logits
        else:
            return source_data, target_data

    else:
        source_batch_size = source_data.size(0)
        target_batch_size = target_data.size(0)
        min_batch_size = min(source_batch_size, target_batch_size)

        if source_batch_size > min_batch_size:
            source_data = source_data[:min_batch_size]
        if target_batch_size > min_batch_size:
            target_data = target_data[:min_batch_size]

        return source_data, target_data


def train(model, source_loader, target_loader, test_loader, save_path='./model_transformer/',
          n_epochs=100, lr=0.001, use_transfer_loss=False, transfer_loss='adv', max_iter=1000,
          source_probs=None, target_labels=None, feature_dim=None, S_T_Num=1.0):
    """训练函数"""
    criterion = nn.CrossEntropyLoss()

    main_params = []
    transfer_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name or 'fc' in name:
            main_params.append(param)
        elif 'adaptor' in name or 'discriminator' in name:
            transfer_params.append(param)
        else:
            main_params.append(param)

    optimizer = optim.AdamW([
        {'params': main_params, 'lr': 5e-4, 'weight_decay': 1e-5},
        {'params': transfer_params, 'lr': 5e-4, 'weight_decay': 0}
    ])

    main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    transfer_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    if use_transfer_loss:
        data, _ = next(iter(target_loader))
        data = model(data, return_intermediate=True)['transformer']
        feature_dim = data.contiguous().view(data.size(0), -1).size(1)
        print('feature_dim:', feature_dim)
        num_class = len(np.unique([label.item() for _, label in source_loader.dataset]))
        transfer_loss_args = {
            "loss_type": transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        adapt_loss = TransferLoss(** transfer_loss_args)

    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []
    if use_transfer_loss:
        transfer_loss_list = []

    train_acc_list.clear()
    train_loss_list.clear()
    test_acc_list.clear()
    test_loss_list.clear()
    if use_transfer_loss:
        transfer_loss_list.clear()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('the model will be trained on: ', device)
    model = model.to(device)

    best_accuracy = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_auc = 0.0
    best_cm = None

    if source_probs is not None and target_labels is not None:
        print("正在按源域真实标签的预测概率抽取样本...")
        target_classes, target_counts = np.unique(target_labels, return_counts=True)
        target_cls_dict = dict(zip(target_classes, target_counts))

        source_data = []
        source_labels = []
        for inputs, labels in source_loader:
            source_data.append(inputs.cpu().numpy())
            source_labels.append(labels.cpu().numpy())
        source_data = np.vstack(source_data)
        source_labels = np.hstack(source_labels)
        source_classes, source_class_counts = np.unique(source_labels, return_counts=True)
        print(f"源域类别分布: {dict(zip(source_classes, source_class_counts))}")

        selected_indices = []
        for cls in source_classes:
            if cls not in target_classes:
                continue

            target_n = int(target_cls_dict[cls] * (S_T_Num))
            print(f"\n处理源域类别 {cls}，目标抽取数量: {target_n}")

            cls_indices = np.where(source_labels == cls)[0]
            cls_size = len(cls_indices)

            if cls_size == 0:
                print(f"  源域类别 {cls} 无样本，跳过")
                continue

            cls_probs = source_probs[cls_indices, cls]
            sorted_indices = np.argsort(-cls_probs)
            available_n = min(cls_size, target_n)
            selected_cls_indices = cls_indices[sorted_indices[:available_n]]

            if available_n > 0:
                print(f"  成功抽取 {available_n} 个样本")
            else:
                print(f"  该类别无可用样本")

            if available_n < target_n:
                print(f"  警告: 源域类别 {cls} 样本数不足（{available_n} < {target_n}）")

            selected_indices.extend(selected_cls_indices)

        selected_source_data = source_data[selected_indices]
        selected_source_labels = source_labels[selected_indices]
        selected_source_dataset = TensorDataset(
            torch.from_numpy(selected_source_data).float().to(device),
            torch.from_numpy(selected_source_labels).long().to(device)
        )
        selected_source_loader = DataLoader(selected_source_dataset, batch_size=32, shuffle=True)

        print("\n抽取的源域样本分布:")
        selected_dist = np.bincount(selected_source_labels, minlength=len(source_classes))
        for cls, count in enumerate(selected_dist):
            print(f"  类别 {cls}: {count} 样本")
    else:
        selected_source_loader = source_loader
        print("使用全部源域样本")

    for epoch in range(n_epochs):
        training_loss = 0.0
        testing_loss = 0.0
        correct = 0
        total = 0
        if use_transfer_loss:
            epoch_transfer_loss = 0.0

        model.train()

        if use_transfer_loss and target_loader is not None:
            source_iter = iter(selected_source_loader)
            target_iter = iter(target_loader)
            num_batches = min(len(selected_source_loader), len(target_loader))
            max_iter = n_epochs * len(selected_source_loader)
            phase1_end = int(0.3 * max_iter)
            phase2_end = int(0.8 * max_iter)

            for batch_idx in tqdm(range(num_batches)):
                try:
                    source_inputs, source_labels = next(source_iter)
                except StopIteration:
                    source_iter = iter(selected_source_loader)
                    source_inputs, source_labels = next(source_iter)

                try:
                    target_inputs, target_labels = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_inputs, target_labels = next(target_iter)

                target_inputs = target_inputs.to(device)
                target_labels = target_labels.to(device)
                source_inputs = source_inputs.to(device)

                optimizer.zero_grad()

                outputs = model(target_inputs)
                source_features = model(source_inputs, return_intermediate=True)['transformer']
                target_features = model(target_inputs, return_intermediate=True)['transformer']

                clf_loss = criterion(outputs, target_labels)
                source_processed, target_processed = preprocess_data_for_loss(
                    source_features, target_features, loss_type=transfer_loss)
                transfer_loss_value = adapt_loss(source_processed, target_processed)

                epoch_transfer_loss += transfer_loss_value.item()

                current_iter = epoch * len(selected_source_loader) + batch_idx
                if current_iter <= phase1_end:
                    lambda_weight = 0.3
                elif current_iter <= phase2_end:
                    lambda_weight = 1.0
                else:
                    lambda_weight = 0.5

                loss = clf_loss + transfer_loss_value * lambda_weight
                loss.backward()
                optimizer.step()

                main_scheduler.step()
                transfer_scheduler.step()

                training_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += target_labels.size(0)
                correct += (predicted == target_labels).sum().item()
        else:
            for inputs, labels in tqdm(selected_source_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                clf_loss = criterion(outputs, labels)
                loss = clf_loss

                loss.backward()
                optimizer.step()
                main_scheduler.step()

                training_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_loss = training_loss / (num_batches if use_transfer_loss else len(selected_source_loader))
        train_loss_list.append(train_loss)
        train_accuracy = correct / total
        train_acc_list.append(train_accuracy)

        if use_transfer_loss:
            avg_transfer_loss = epoch_transfer_loss / num_batches
            transfer_loss_list.append(avg_transfer_loss)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            all_probs = []

            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                testing_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())

            test_loss = testing_loss / len(test_loader)
            test_loss_list.append(test_loss)
            test_accuracy = correct / total
            test_acc_list.append(test_accuracy)

            cm = confusion_matrix(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            if len(np.unique(all_labels)) == 2:
                y_score = np.array(all_probs)[:, 1]
                auc = roc_auc_score(all_labels, y_score)
            else:
                y_score = np.array(all_probs)
                auc = roc_auc_score(all_labels, y_score, multi_class='ovr', average='macro')

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_precision = precision
                best_recall = recall
                best_f1 = f1
                best_auc = auc
                best_cm = cm
                if save_path is not None:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(model.state_dict(), save_path + 'best_model.pth')
                    print("best_model found, best acc: ", best_accuracy)

        if use_transfer_loss:
            print(
                f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.4f} - Transfer Loss: {avg_transfer_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
        else:
            print(
                f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")

    if save_path is not None and os.path.exists(save_path + 'best_model.pth'):
        model.load_state_dict(torch.load(save_path + 'best_model.pth'))

    return model, best_accuracy, best_precision, best_recall, best_f1, best_auc, best_cm


def record_source_probs(model, source_loader):
    """记录源域样本预测概率"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(source_loader):
            inputs = inputs.to(device)

            outputs = model(inputs)

            probs = F.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    return all_probs


def runModel(X_pretrain, Y_pretrain, X_train, Y_train, X_test, Y_test, channel_num, transfer_loss='adv', S_T_Num=1.0):
    """模型运行流程"""
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    n_outputs = len(np.unique(Y_train))

    model = EEGConformer(
        n_outputs=n_outputs,
        n_chans=channel_num,
        n_times=X_train.shape[2],
        input_window_seconds=3.0,
        sfreq=250,
    )

    Y_pretrain = Y_pretrain.astype(np.int64)
    Y_train = Y_train.astype(np.int64)
    Y_test = Y_test.astype(np.int64)

    x_pretrain_tensor = torch.from_numpy(X_pretrain).to(torch.float32).to(device)
    y_pretrain_tensor = torch.from_numpy(Y_pretrain).to(torch.long).to(device)
    pretrain_dataset = TensorDataset(x_pretrain_tensor, y_pretrain_tensor)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True)

    x_train_tensor = torch.from_numpy(X_train).to(torch.float32).to(device)
    y_train_tensor = torch.from_numpy(Y_train).to(torch.long).to(device)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    x_test_tensor = torch.from_numpy(X_test).to(torch.float32).to(device)
    y_test_tensor = torch.from_numpy(Y_test).to(torch.long).to(device)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("开始预训练模型...")
    # pretrained_model, _, _, _, _, _, _ = train(model, pretrain_loader, None, test_loader,
    #                                            save_path='./pretrained_model1/', n_epochs=70,
    #                                            use_transfer_loss=False)
    #
    # print("记录源域样本的预测概率...")
    # source_probs = record_source_probs(pretrained_model, pretrain_loader)
    # 加载已保存的模型权重
    pretrained_model = model
    pretrained_model_path = './pretrained_model1/best_model.pth'
    if os.path.exists(pretrained_model_path):
        pretrained_model.load_state_dict(torch.load(pretrained_model_path))
        print(f"成功加载预训练模型：{pretrained_model_path}")
    else:
        raise FileNotFoundError(f"未找到预训练模型文件：{pretrained_model_path}")

    print("记录源域样本的预测概率...")
    source_probs = record_source_probs(pretrained_model, pretrain_loader)
    print("开始微调模型...")
    fine_tune_save_path = './fine_tuned_model3/'

    best_model, acc, precision, recall, f1, auc, cm = train(pretrained_model, pretrain_loader, train_loader,
                                                            test_loader, S_T_Num=S_T_Num,
                                                            save_path=fine_tune_save_path, n_epochs=100, lr=0.0001,
                                                            use_transfer_loss=False,
                                                            transfer_loss=transfer_loss,
                                                            max_iter=100 * len(train_loader),
                                                            source_probs=source_probs,
                                                            feature_dim=X_train.shape[2],
                                                            target_labels=Y_train)

    return acc, precision, recall, f1, auc, cm


if __name__ == '__main__':
    # 超参数
    window_size = 3.0
    window_size1 = 3.0
    step_size = 0.5
    voting = 'hard_median'
    sfreq = 250
    sub_num = 12  # 被试数量
    start_num = 12  # 起始被试编号

    transfer_loss_type = 'mmd'  # 迁移损失类型

    if not os.path.exists('Result'):
        os.makedirs('Result')

    all_results = []
    sub_num1 = 25
    start_num1 = 13

    for subj in range(start_num, sub_num + 1):
        print(f"\n===== 开始处理受试者 {subj} =====")
        # 加载数据
        GroupData, GroupLabel = loadSourceData(sub_num1, start_num1)
        IndividualData, IndividualLabel = loadTargetData(subj)

        # 计算通道权重
        GroupChannelWeight = []
        for GroupMember in range(1, sub_num1 - start_num1 + 1):
            MemberData = GroupData[(GroupMember - 1) * 450:GroupMember * 450]
            print(f'Calculating channel weight for group member {GroupMember}...')
            GroupChannelWeight.append(channel_weight_calculation(MemberData, sfreq, window_size, step_size, voting))
        GroupChannelWeight = np.array(GroupChannelWeight)
        GroupChannelWeight = np.sum(GroupChannelWeight, axis=0) / (sub_num1 - start_num1)
        print(f"组通道权重形状: {GroupChannelWeight.shape}")

        IndividualChannelWeight = channel_weight_calculation(IndividualData, sfreq, window_size, step_size, voting)
        print(f"个体通道权重形状: {IndividualChannelWeight.shape}")

        # 遍历alpha值
        for alpha in [0.4]:
            print(f"\n----- 使用alpha = {alpha} -----")
            start_time = time.time()

            # 计算混合通道权重
            MixedChannelWeight = alpha * GroupChannelWeight + (1 - alpha) * IndividualChannelWeight

            # 对混合通道权重进行排序，获取排序后的索引
            sorted_channels = np.argsort(-MixedChannelWeight.flatten())

            # 遍历每个通道数
            for channel_num in [12]:
                print(f"\n处理通道数: {channel_num}")
                # 选择通道
                selected_GroupData = []
                selected_IndividualData = []
                for i in range(channel_num):
                    selected_GroupData.append(GroupData[:, sorted_channels[i], :])
                    selected_IndividualData.append(IndividualData[:, sorted_channels[i], :])
                selected_GroupData = np.transpose(np.squeeze(np.array(selected_GroupData)), [1, 0, 2])
                selected_IndividualData = np.transpose(np.squeeze(np.array(selected_IndividualData)), [1, 0, 2])

                # 1. 提取选中通道的权重（按排序后的通道索引取对应权重）
                selected_weights = MixedChannelWeight[sorted_channels[:channel_num]]
                # 2. 权重归一化（使用Softmax确保权重和为1，避免数值过大）
                selected_weights = np.exp(selected_weights) / np.sum(np.exp(selected_weights))  # Softmax归一化
                # 3. 对选中的源域和目标域数据按通道加权（逐通道乘以归一化后的权重）
                weighted_GroupData = selected_GroupData * selected_weights[np.newaxis, :, np.newaxis]
                weighted_IndividualData = selected_IndividualData * selected_weights[np.newaxis, :, np.newaxis]
                # ------------------------------------------------------

                # 选择固定窗口
                window_start = 0
                window_end = window_start + int(window_size1 * sfreq)
                window = (window_start, window_end)
                print(f"使用窗口: {window}")

                # 获取窗口数据（使用加权后的通道数据）
                window_group_data = weighted_GroupData[:, :, window[0]:window[1]]
                window_individual_data = weighted_IndividualData[:, :, window[0]:window[1]]

                # 划分数据集
                X_pretrain, Y_pretrain, X_train, Y_train, X_test, Y_test = shuffleDataset_Junheng(
                    window_group_data, GroupLabel, window_individual_data, IndividualLabel)

                # 遍历S_T比例并训练
                for S_T in [1.0]:
                    acc, precision, recall, f1, auc, cm = runModel(
                        X_pretrain, Y_pretrain, X_train, Y_train, X_test, Y_test,
                        channel_num, transfer_loss=transfer_loss_type, S_T_Num=S_T
                    )

                    current_result = {
                        '受试者编号': subj,
                        'alpha值': alpha,
                        'S_T_Num': S_T,
                        '通道数': channel_num,
                        '所选通道索引': list(sorted_channels[:channel_num]),
                        '通道权重（归一化后）': selected_weights.tolist(),
                        '窗口': window,
                        '准确率': acc,
                        '精确率': precision,
                        '召回率': recall,
                        'F1分数': f1,
                        'AUC': auc,
                        '混淆矩阵': cm.tolist(),
                        '训练时间(秒)': time.time() - start_time,
                        '特征自适应损失类型': transfer_loss_type
                    }

                    all_results.append(current_result)
                    print(f"通道数 {channel_num} 结果已记录 - 准确率: {acc:.4f}, F1: {f1:.4f}")

                    # 保存结果
                    result_df = pd.DataFrame(all_results)
                    excel_path = os.path.join('Result', 'all_subjects_detailed_results_CA5.xlsx')
                    csv_path = os.path.join('Result', 'all_subjects_detailed_results_CA5.csv')

                    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                        result_df.to_excel(writer, index=False, sheet_name='DetailedResults')
                    result_df.to_csv(csv_path, index=False)

            print(f"已保存当前结果至 {excel_path} 和 {csv_path}")

    print("\n所有被试处理完成！")
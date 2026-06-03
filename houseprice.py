"""Kaggle 房价预测示例。

本文件保留 D2L 第 4 章的房价预测流程，但使用仓库内的 `mini_d2l.py`
提供数据加载和绘图工具，不依赖第三方 `d2l` 包。
"""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn

import mini_d2l as d2l


d2l.DATA_HUB["kaggle_house_pred_train"] = (
    d2l.DATA_URL + "kaggle_house_pred_train.csv",
    "585e9cc93e70b39160e7921475f9bcd7d31219ce",
)

d2l.DATA_HUB["kaggle_house_pred_test"] = (
    d2l.DATA_URL + "kaggle_house_pred_test.csv",
    "fa19780a7b011d9b009e8bff8e99922a8ee2eb90",
)


def load_house_data():
    """下载并读取 Kaggle 房价预测训练集和测试集。"""
    train_data = pd.read_csv(d2l.download("kaggle_house_pred_train"))
    test_data = pd.read_csv(d2l.download("kaggle_house_pred_test"))
    return train_data, test_data


def preprocess_house_features(train_data, test_data):
    """标准化数值特征并对类别特征做 one-hot 编码。"""
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / x.std())
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=True).astype(float)

    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
    return train_features, test_features, train_labels


def get_net(num_inputs: int):
    """创建线性回归模型。"""
    return nn.Sequential(nn.Linear(num_inputs, 1))


def log_rmse(net, features, labels):
    """计算 Kaggle 房价任务常用的 log RMSE。"""
    loss = nn.MSELoss()
    clipped_preds = torch.clamp(net(features), 1, float("inf"))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    """训练模型并返回训练集与验证集的 log RMSE 曲线。"""
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size) if test_features is not None else None
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss = nn.MSELoss()

    for _ in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_iter is not None and test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    """返回第 i 折交叉验证的训练和验证数据。"""
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    X_valid, y_valid = None, None

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)

    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    """执行 k 折交叉验证并返回平均训练和验证 log RMSE。"""
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(
                list(range(1, num_epochs + 1)),
                [train_ls, valid_ls],
                xlabel="epoch",
                ylabel="rmse",
                xlim=[1, num_epochs],
                legend=["train", "valid"],
                yscale="log",
            )
        print(f"折{i + 1}，训练log rmse {float(train_ls[-1]):f}，验证log rmse {float(valid_ls[-1]):f}")

    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, learning_rate, weight_decay, batch_size):
    """训练最终模型并导出 Kaggle 提交文件。"""
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, learning_rate, weight_decay, batch_size)
    d2l.plot(
        list(range(1, num_epochs + 1)),
        [train_ls],
        xlabel="epoch",
        ylabel="log rmse",
        xlim=[1, num_epochs],
        yscale="log",
    )
    print(f"训练log rmse：{float(train_ls[-1]):f}")

    preds = net(test_features).detach().numpy()
    test_data["SalePrice"] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data["Id"], test_data["SalePrice"]], axis=1)
    submission.to_csv("submission.csv", index=False)


def main():
    """运行完整 Kaggle 房价预测示例。"""
    train_data, test_data = load_house_data()
    train_features, test_features, train_labels = preprocess_house_features(train_data, test_data)
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    print(f"{k}-折验证: 平均训练log rmse: {float(train_l):f}, 平均验证log rmse: {float(valid_l):f}")
    train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)


if __name__ == "__main__":
    main()

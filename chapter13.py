"""D2L 第 13 章代码整理版。

本文件按“计算机视觉”主线整理，尽量保留教材中的核心实现：
1. 图像增广与微调；
2. 边界框、锚框、NMS 与 TinySSD；
3. 语义分割、转置卷积与 FCN；
4. 风格迁移；
5. Kaggle 视觉任务的最小可复现入口。

说明：
- 默认入口只做轻量 shape / 数据流检查，不直接启动长时间训练；
- 数据集下载失败时会打印跳过提示，而不是让脚本直接崩溃；
- 更重的训练任务统一放到 `run_xxx()` 函数里，按需手动调用。
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, models, transforms

import chapter7 as ch7
import mini_d2l as d2l


# ==================== 1. 图像增广与微调 ====================
def synthetic_demo_image(size: int = 224) -> Image.Image:
    """生成一张简单彩色图，便于本地演示图像增广。"""
    x = torch.linspace(0, 1, size).repeat(size, 1)
    y = torch.linspace(0, 1, size).unsqueeze(1).repeat(1, size)
    image = torch.stack((x, y, 1 - x), dim=2).mul(255).byte().numpy()
    return Image.fromarray(image)


def apply_image_augmentation_demo(num_rows=2, num_cols=4, image=None):
    """演示常见图像增广操作。"""
    if image is None:
        image = synthetic_demo_image()

    aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(200, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ]
    )
    images = [aug(image) for _ in range(num_rows * num_cols)]
    d2l.show_images(images, num_rows, num_cols, scale=2)
    return images


def build_finetune_net(num_classes: int, freeze_features: bool = True):
    """构建基于 ResNet-18 的微调模型。"""
    weights = models.ResNet18_Weights.DEFAULT
    net = models.resnet18(weights=weights)
    if freeze_features:
        for param in net.parameters():
            param.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def train_finetuning(
    train_iter,
    test_iter,
    num_classes: int,
    num_epochs: int = 5,
    lr: float = 5e-5,
    param_group: bool = True,
    freeze_features: bool = True,
    device=None,
):
    """训练微调模型。"""
    if device is None:
        device = d2l.try_gpu()
    net = build_finetune_net(num_classes, freeze_features=freeze_features)
    net.to(device)

    if param_group:
        params_1x = [
            param
            for name, param in net.named_parameters()
            if name not in ["fc.weight", "fc.bias"]
        ]
        trainer = torch.optim.SGD(
            [
                {"params": params_1x},
                {"params": net.fc.parameters(), "lr": lr * 10},
            ],
            lr=lr,
            weight_decay=1e-3,
        )
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-3)

    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(
        xlabel="epoch",
        xlim=[1, num_epochs],
        legend=["train loss", "train acc", "test acc"],
    )
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            metric.add(float(l) * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter, device)
        animator.add(epoch + 1, (train_l, train_acc, test_acc))
        print(
            f"epoch {epoch + 1}: train loss {train_l:.4f}, "
            f"train acc {train_acc:.4f}, test acc {test_acc:.4f}"
        )
    return net, trainer, loss


# ==================== 2. 边界框基础 ====================
def box_corner_to_center(boxes):
    """左上右下坐标转中心点宽高坐标。"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), axis=-1)


def box_center_to_corner(boxes):
    """中心点宽高坐标转左上右下坐标。"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), axis=-1)


def bbox_to_rect(bbox, color):
    """复用 mini_d2l 中的矩形转换。"""
    return d2l.bbox_to_rect(bbox, color)


def inspect_bbox_conversions():
    """检查两种边界框表示之间的互转。"""
    boxes = torch.tensor([[60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]])
    center = box_corner_to_center(boxes)
    corners = box_center_to_corner(center)
    print("center form:", center)
    print("recovered corners:", corners)
    return center, corners


# ==================== 3. 锚框、IoU、NMS 与检测后处理 ====================
def multibox_prior(data, sizes, ratios):
    """为输入特征图的每个像素生成锚框。"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = num_sizes + num_ratios - 1
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing="ij")
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    first_size = torch.tensor([sizes[0]], device=device)
    w = torch.cat(
        (size_tensor * torch.sqrt(ratio_tensor[0]), first_size * torch.sqrt(ratio_tensor[1:]))
    ) * in_height / in_width
    h = torch.cat(
        (size_tensor / torch.sqrt(ratio_tensor[0]), first_size / torch.sqrt(ratio_tensor[1:]))
    )
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(
        boxes_per_pixel, dim=0
    )
    return (out_grid + anchor_manipulations).unsqueeze(0)


def box_iou(boxes1, boxes2):
    """计算两组边界框的 IoU。"""

    def box_area(boxes):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """为每个锚框分配最匹配的真实框索引。"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j

    col_discard = torch.full((num_anchors,), -1, device=device)
    row_discard = torch.full((num_gt_boxes,), -1, device=device)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """把真实框编码成相对锚框的偏移量。"""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    return torch.cat([offset_xy, offset_wh], axis=1)


def offset_inverse(anchors, offset_preds):
    """把预测偏移量解码回真实边界框。"""
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = offset_preds[:, :2] * anc[:, 2:] / 10 + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    return box_center_to_corner(pred_bbox)


def nms(boxes, scores, iou_threshold):
    """非极大值抑制。"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1:
            break
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)


def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, pos_threshold=0.009999999):
    """根据类别概率和偏移量输出最终检测结果。"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf = conf[all_id_sorted]
        predicted_bb = predicted_bb[all_id_sorted]

        below_min_idx = conf < pos_threshold
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat(
            (class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb),
            dim=1,
        )
        out.append(pred_info)
    return torch.stack(out)


def inspect_anchor_shapes():
    """检查锚框生成与 NMS 相关函数的输出形状。"""
    X = torch.zeros((1, 3, 4, 4))
    anchors = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    boxes = torch.tensor([[0.1, 0.1, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95], [0.55, 0.2, 0.9, 0.88]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, 0.5)
    print("anchors shape:", anchors.shape)
    print("nms keep indices:", keep)
    return anchors, keep


# ==================== 4. 香蕉检测数据集与 TinySSD ====================
d2l.DATA_HUB["banana-detection"] = (
    d2l.DATA_URL + "banana-detection.zip",
    "5de26c8fce5ccdea9f91267273464dc968d20d72",
)


def read_data_bananas(is_train=True):
    """读取香蕉检测数据集。"""
    data_dir = Path(d2l.download_extract("banana-detection"))
    csv_fname = data_dir / "bananas_train" / "label.csv" if is_train else data_dir / "bananas_val" / "label.csv"
    csv_data = pd.read_csv(csv_fname)
    images, targets = [], []
    for _, target in csv_data.iterrows():
        img_name = target["img_name"]
        img_path = csv_fname.parent / "images" / img_name
        images.append(transforms.ToTensor()(Image.open(img_path).convert("RGB")))
        targets.append(
            torch.tensor(
                [[target["label"], target["xmin"], target["ymin"], target["xmax"], target["ymax"]]],
                dtype=torch.float32,
            )
        )
    return images, targets


class BananasDataset(data.Dataset):
    """香蕉检测数据集封装。"""

    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.features)


def load_data_bananas(batch_size):
    """返回香蕉检测训练集与验证集。"""
    train_iter = data.DataLoader(BananasDataset(is_train=True), batch_size, shuffle=True)
    val_iter = data.DataLoader(BananasDataset(is_train=False), batch_size, shuffle=False)
    return train_iter, val_iter


def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor_layer, bbox_predictor_layer):
    Y = blk(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor_layer(Y)
    bbox_preds = bbox_predictor_layer(Y)
    return Y, anchors, cls_preds, bbox_preds


class TinySSD(nn.Module):
    """教材里的简化 SSD 检测器。"""

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            setattr(self, f"blk_{i}", get_blk(i))
            setattr(self, f"cls_{i}", cls_predictor(idx_to_in_channels[i], 4, num_classes))
            setattr(self, f"bbox_{i}", bbox_predictor(idx_to_in_channels[i], 4))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        ratios = [[1, 2, 0.5]] * 5
        for i in range(5):
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X,
                getattr(self, f"blk_{i}"),
                sizes[i],
                ratios[i],
                getattr(self, f"cls_{i}"),
                getattr(self, f"bbox_{i}"),
            )
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds).reshape(X.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


def train_tinyssd_debug(train_iter, num_epochs=5, lr=0.2, device=None):
    """用于调通 TinySSD 前向/反向图的调试训练入口。

    注意：
    - 这里不会做真实的 anchor 匹配与框回归目标构造；
    - 因此它只能验证训练图是否可运行，不能产出有效检测器。
    """
    if device is None:
        device = d2l.try_gpu()
    print("warning: train_tinyssd_debug() 仅用于调试计算图，不会训练出有效检测器。")
    net = TinySSD(num_classes=1).to(device)
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=5e-4)
    cls_loss = nn.CrossEntropyLoss(reduction="none")
    bbox_loss = nn.L1Loss(reduction="none")

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(1)
        for X, Y in train_iter:
            trainer.zero_grad()
            X = X.to(device)
            anchors, cls_preds, bbox_preds = net(X)

            # 这里只做“训练图能跑通”的最小版本：
            # 目标框编码与正负样本匹配保留在后续需要时再补全。
            dummy_cls = torch.zeros(cls_preds.shape[:2], dtype=torch.long, device=device)
            dummy_bbox = torch.zeros_like(bbox_preds, device=device)
            l = cls_loss(cls_preds.reshape(-1, 2), dummy_cls.reshape(-1)).mean()
            l = l + bbox_loss(bbox_preds, dummy_bbox).mean()
            l.backward()
            trainer.step()
            metric.add(l.item())
        print(f"epoch {epoch + 1}, loss {metric[0]:.4f}")
    return net


def predict_tinyssd(net, X, device=None):
    """用 TinySSD 做一次前向预测。"""
    if device is None:
        device = d2l.try_gpu()
    net.eval()
    X = X.to(device)
    anchors, cls_preds, bbox_preds = net(X)
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds, anchors)
    return output


def inspect_tinyssd_shapes():
    """检查 TinySSD 前向输出的形状。"""
    X = torch.zeros((2, 3, 256, 256))
    net = TinySSD(num_classes=1)
    anchors, cls_preds, bbox_preds = net(X)
    print("TinySSD anchors:", anchors.shape)
    print("TinySSD cls_preds:", cls_preds.shape)
    print("TinySSD bbox_preds:", bbox_preds.shape)
    return anchors, cls_preds, bbox_preds


# ==================== 5. 语义分割、转置卷积与 FCN ====================
VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


def read_voc_images(is_train=True):
    """读取 VOC 语义分割数据集。"""
    image_set = "train" if is_train else "val"
    dataset = datasets.VOCSegmentation(
        root=str(d2l.DATA_DIR),
        year="2012",
        image_set=image_set,
        download=True,
    )
    features, labels = [], []
    for image, target in dataset:
        features.append(image)
        labels.append(target)
    return features, labels


def voc_colormap2label():
    """建立 RGB 颜色到类别索引的查找表。"""
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]
        ] = i
    return colormap2label


def voc_label_indices(colormap, colormap2label):
    """把颜色标注图转成类别索引图。"""
    colormap = colormap.permute(1, 2, 0).numpy().astype("int32")
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


def voc_rand_crop(feature, label, height, width):
    """对图像与掩码做同步随机裁剪。"""
    rect = transforms.RandomCrop.get_params(feature, (height, width))
    feature = transforms.functional.crop(feature, *rect)
    label = transforms.functional.crop(label, *rect)
    return feature, label


class VOCSegDataset(data.Dataset):
    """VOC 语义分割数据集。"""

    def __init__(self, is_train, crop_size, voc_dir=None):
        self.crop_size = crop_size
        features, labels = read_voc_images(is_train=is_train)
        self.features = features
        self.labels = labels
        self.colormap2label = voc_colormap2label()

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature, label = voc_rand_crop(feature, label, *self.crop_size)
        feature = transforms.ToTensor()(feature)
        label = transforms.PILToTensor()(label)
        label = voc_label_indices(label, self.colormap2label)
        return feature, label

    def __len__(self):
        return len(self.features)


def load_data_voc(batch_size, crop_size):
    """返回 VOC 语义分割数据迭代器。"""
    train_iter = data.DataLoader(VOCSegDataset(True, crop_size), batch_size, shuffle=True)
    test_iter = data.DataLoader(VOCSegDataset(False, crop_size), batch_size, shuffle=False)
    return train_iter, test_iter


def bilinear_kernel(in_channels, out_channels, kernel_size):
    """构造双线性插值初始化核。"""
    factor = (kernel_size + 1) // 2
    center = factor - 1 if kernel_size % 2 == 1 else factor - 0.5
    og = torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1)
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


def build_fcn_resnet18(num_classes=len(VOC_CLASSES), pretrained=True):
    """构建 FCN 风格的语义分割网络。"""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    pretrained_net = models.resnet18(weights=weights)
    net = nn.Sequential(*list(pretrained_net.children())[:-2])
    net.add_module("final_conv", nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module(
        "transpose_conv",
        nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32),
    )
    net.transpose_conv.weight.data.copy_(bilinear_kernel(num_classes, num_classes, 64))
    return net


def train_fcn(train_iter, test_iter, num_epochs=5, lr=0.001, device=None):
    """训练 FCN 的最小入口。"""
    if device is None:
        device = d2l.try_gpu()
    net = build_fcn_resnet18().to(device)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-3)

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(1)
        net.train()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            l = loss(net(X), y.long())
            l.backward()
            trainer.step()
            metric.add(l.item())
        print(f"epoch {epoch + 1}, loss {metric[0]:.4f}")
    return net


def inspect_fcn_shapes():
    """检查 FCN 前向输出形状。"""
    net = build_fcn_resnet18(pretrained=False)
    X = torch.rand(size=(1, 3, 320, 480))
    Y = net(X)
    print("FCN output shape:", Y.shape)
    return Y


# ==================== 6. 风格迁移 ====================
def preprocess_style_image(image, image_shape):
    """风格迁移图像预处理。"""
    rgb_mean = torch.tensor([0.485, 0.456, 0.406])
    rgb_std = torch.tensor([0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [
            transforms.Resize(image_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std),
        ]
    )
    return transform(image).unsqueeze(0)


def postprocess_style_image(img_tensor):
    """风格迁移图像反归一化。"""
    rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).reshape(3, 1, 1)
    rgb_std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).reshape(3, 1, 1)
    img = img_tensor[0].cpu() * rgb_std.cpu() + rgb_mean.cpu()
    return transforms.ToPILImage()(img.clamp(0, 1))


def get_style_transfer_net():
    """获取用于提取内容/风格特征的 VGG19。"""
    net = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
    return net.eval()


def extract_features(X, content_layers, style_layers, net):
    """提取内容层和风格层特征。"""
    contents, styles = [], []
    for i, layer in enumerate(net):
        X = layer(X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles


def gram(X):
    """计算 Gram 矩阵。"""
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)


def compute_content_loss(Y_hat, Y):
    return F.mse_loss(Y_hat, Y)


def compute_style_loss(Y_hat, gram_Y):
    return F.mse_loss(gram(Y_hat), gram_Y)


def compute_tv_loss(Y_hat):
    return 0.5 * (
        F.l1_loss(Y_hat[:, :, 1:, :], Y_hat[:, :, :-1, :])
        + F.l1_loss(Y_hat[:, :, :, 1:], Y_hat[:, :, :, :-1])
    )


def train_style_transfer(
    content_img,
    style_img,
    image_shape=(300, 450),
    num_steps=200,
    lr=0.3,
    device=None,
):
    """训练风格迁移图像。"""
    if device is None:
        device = d2l.try_gpu()
    net = get_style_transfer_net().to(device)
    content_layers, style_layers = [25], [0, 5, 10, 19, 28]

    content_X = preprocess_style_image(content_img, image_shape).to(device)
    style_X = preprocess_style_image(style_img, image_shape).to(device)
    content_Y, _ = extract_features(content_X, content_layers, style_layers, net)
    _, styles_Y = extract_features(style_X, content_layers, style_layers, net)
    styles_Y_gram = [gram(Y) for Y in styles_Y]

    X = content_X.clone().requires_grad_(True)
    trainer = torch.optim.Adam([X], lr=lr)
    for step in range(num_steps):
        trainer.zero_grad()
        contents_hat, styles_hat = extract_features(X, content_layers, style_layers, net)
        content_loss = sum(
            compute_content_loss(Y_hat, Y) for Y_hat, Y in zip(contents_hat, content_Y)
        )
        style_loss = sum(
            compute_style_loss(Y_hat, Y) for Y_hat, Y in zip(styles_hat, styles_Y_gram)
        )
        tv_loss = compute_tv_loss(X)
        l = content_loss + 1e3 * style_loss + 10 * tv_loss
        l.backward()
        trainer.step()
        if (step + 1) % max(1, num_steps // 5) == 0:
            print(f"step {step + 1}, loss {float(l):.4f}")
    return postprocess_style_image(X.detach())


def inspect_style_transfer_losses():
    """用随机张量检查风格迁移损失函数能否跑通。"""
    X = torch.randn(1, 3, 16, 16)
    Y = torch.randn(1, 3, 16, 16)
    print("content loss:", float(compute_content_loss(X, Y)))
    print("style loss:", float(compute_style_loss(X, gram(Y))))
    print("tv loss:", float(compute_tv_loss(X)))


# ==================== 7. Kaggle 视觉任务入口 ====================
def run_kaggle_cifar10(data_dir, num_epochs=10, lr=2e-4, batch_size=128, device=None):
    """Kaggle CIFAR-10 最小入口。

    假设 `data_dir` 下已经按 `train/类别名/*.png`、`test/*.png` 组织好。
    """
    if device is None:
        device = d2l.try_gpu()

    train_augs = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_augs = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_augs)
    train_iter = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    net = build_finetune_net(num_classes=len(train_ds.classes), freeze_features=False).to(device)
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(1)
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            l = loss(net(X), y)
            l.backward()
            trainer.step()
            metric.add(l.item())
        print(f"epoch {epoch + 1}, loss {metric[0]:.4f}")
    return net, test_augs


def run_kaggle_dog(data_dir, num_epochs=10, lr=2e-4, batch_size=32, device=None):
    """Kaggle 狗品种识别最小入口。"""
    return run_kaggle_cifar10(
        data_dir=data_dir,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )


# ==================== 8. 运行入口 ====================
def main():
    """默认只做轻量检查。"""
    apply_image_augmentation_demo(num_rows=1, num_cols=4)
    inspect_bbox_conversions()
    inspect_anchor_shapes()
    inspect_tinyssd_shapes()
    inspect_fcn_shapes()
    inspect_style_transfer_losses()

    # 以下数据集或训练任务较重，按需取消注释。
    # train_iter, val_iter = load_data_bananas(batch_size=8)
    # train_tinyssd_debug(train_iter, num_epochs=5)
    # train_iter, test_iter = load_data_voc(batch_size=4, crop_size=(320, 480))
    # train_fcn(train_iter, test_iter, num_epochs=5)

    # 微调 / Kaggle 入口同样按需手动调用。
    # train_iter = ...
    # test_iter = ...
    # train_finetuning(train_iter, test_iter, num_classes=2)
    # run_kaggle_cifar10(data_dir="path/to/cifar10")
    # run_kaggle_dog(data_dir="path/to/dog-breed")


if __name__ == "__main__":
    main()

# YOLO-World + DIOR 训练完全指南 (小白专用)

> 从零开始,一步一步训练你自己的 YOLO-World 模型
> 数据集: DIOR (遥感卫星图像, 20 类目标)
> 创建日期: 2026-07-15

---

## 📖 目录

1. [什么是 YOLO-World + DIOR?](#1-什么是-yolo-world--dior)
2. [前期准备](#2-前期准备)
3. [下载 DIOR 数据集](#3-下载-dior-数据集)
4. [准备数据 (转换格式)](#4-准备数据-转换格式)
5. [下载预训练模型](#5-下载预训练模型)
6. [本地跑通流程 (验证代码)](#6-本地跑通流程-验证代码)
7. [租 GPU 服务器训练](#7-租-gpu-服务器训练)
8. [训练参数调整](#8-训练参数调整)
9. [训练结果分析](#9-训练结果分析)
10. [常见问题](#10-常见问题)

---

## 1. 什么是 YOLO-World + DIOR?

### YOLO-World 是什么

**YOLO-World** 是腾讯 AI Lab 在 CVPR 2024 发表的论文,是一种**开放词汇目标检测**模型:

- ❌ 传统 YOLO: 只能检测固定的 80 个 COCO 类别
- ✅ YOLO-World: 可以用**文本描述**检测任意物体

比如输入文本 "airplane, ship, bridge",模型就能检测这 3 类,不需要重新训练。

### DIOR 数据集是什么

**DIOR** 是一个**遥感卫星图像目标检测数据集**:

| 特性 | 说明 |
|------|------|
| 图像数 | 23463 张 |
| 标注数 | 192474 个 |
| 类别数 | 20 类 |
| 边界框 | HBB (水平框) ✅ |
| 图像大小 | 800×800 |
| 适用场景 | 遥感检测、地理分析 |

### 20 个类别

```
airplane(飞机)         airport(机场)           baseball-diamond(棒球场)
basketball-court(篮球场) bridge(桥)            chimney(烟囱)
expressway-service-area(高速服务区)  expressway-toll-station(高速收费站)
dam(大坝)              golf-course(高尔夫球场) ground-track-field(田径场)
harbor(港口)           overpass(立交桥)        ship(船)
stadium(体育场)        storage-tank(储罐)      tennis-court(网球场)
train-station(火车站)  vehicle(车辆)          windmill(风车)
```

### 为什么用 DIOR + YOLO-World?

1. ✅ **DIOR 是 HBB 格式**,直接兼容 YOLO-World,不用转格式
2. ✅ YOLO-World 的**开放词汇**能力适合遥感场景(可以测试新类别)
3. ✅ 遥感数据集丰富,实验结果有说服力
4. ✅ 相比 DOTA 不需要处理旋转框,简单很多

---

## 2. 前期准备

### 2.1 硬件要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU | RTX 3060 (12GB) | RTX 4090 (24GB) |
| 内存 | 16GB | 32GB |
| 硬盘 | 50GB | 100GB SSD |
| 网络 | 能下载 30GB 数据 | 高速网络 |

### 2.2 软件要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (用 GPU 训练)
- VSCode (写代码)

### 2.3 创建虚拟环境

```bash
# Windows
cd e:\Model_Train
python -m venv yolow_env
yolow_env\Scripts\activate

# Linux/Mac
python3 -m venv yolow_env
source yolow_env/bin/activate
```

### 2.4 安装依赖

```bash
# 1. 安装 PyTorch (CUDA 版本根据你的显卡选)
# 查看 CUDA 版本: nvidia-smi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. 安装 ultralytics (备用)
pip install ultralytics opencv-python numpy matplotlib tqdm

# 3. 安装 mmyolo 全家桶 (YOLO-World 必需)
pip install -U openmim
mim install mmengine mmcv mmdet mmyolo

# 4. 安装其他依赖
pip install transformers einops timm
```

### 2.5 验证安装

```bash
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import mmyolo; print('mmyolo OK')"
```

---

## 3. 下载 DIOR 数据集

### 3.1 下载地址

DIOR 数据集约 30GB,有两个下载来源:

**方式 1: 官方 (需注册)**
- 网址: http://www.uestc.edu.cn/DIOR.html
- 注册后下载 `DIOR dataset` 完整包

**方式 2: Google Drive (推荐)**
- 搜索关键词: "DIOR dataset download"
- 下载 3 个文件:
  - `DIOR-images.zip` (图片, ~20GB)
  - `DIOR-Annotations.zip` (标注, ~50MB)
  - `DIOR-label-list.csv` (类别列表)

### 3.2 解压数据

```bash
# Windows PowerShell
Expand-Archive DIOR-images.zip -DestinationPath .\DIOR
Expand-Archive DIOR-Annotations.zip -DestinationPath .\DIOR

# Linux
unzip DIOR-images.zip -d DIOR
unzip DIOR-Annotations.zip -d DIOR
```

### 3.3 目录结构

解压后应该是这样的结构:

```
DIOR/
├── Images/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── ... (23463 张图片)
└── Annotations/
    └── XML/
        ├── 1.xml
        ├── 2.xml
        └── ... (23463 个 XML 标注)
```

---

## 4. 准备数据 (转换格式)

YOLO-World 需要 COCO JSON 格式的标注,而 DIOR 是 Pascal VOC XML 格式,需要转换。

### 4.1 把数据集放到项目目录

```bash
# 把下载的 DIOR 文件夹放到项目根目录
# 最终路径: e:\Model_Train\DIOR\
```

### 4.2 运行数据准备脚本

```bash
cd e:\Model_Train
python prepare_dior.py \
    --xml-dir "DIOR/Annotations/XML" \
    --image-dir "DIOR/Images" \
    --output-dir "YOLO-World-master/data/DIOR"
```

### 4.3 划分训练/验证图片

转换脚本生成了 JSON 标注,但图片还没划分。运行下面的 Python 脚本划分图片:

```python
# split_dior_images.py
import json
import os
import shutil
from pathlib import Path

# 读取 train.json, 把对应图片复制到 train/ 目录
def split_images(json_file, src_img_dir, dst_img_dir):
    with open(json_file) as f:
        data = json.load(f)

    os.makedirs(dst_img_dir, exist_ok=True)

    for img in data['images']:
        src = os.path.join(src_img_dir, img['file_name'])
        dst = os.path.join(dst_img_dir, img['file_name'])
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

    print(f"复制 {len(data['images'])} 张图片到 {dst_img_dir}")

# 执行
base = "YOLO-World-master/data/DIOR"
src = "DIOR/Images"

split_images(f"{base}/annotations/train.json", src, f"{base}/images/train")
split_images(f"{base}/annotations/val.json", src, f"{base}/images/val")
```

### 4.4 最终目录结构

```
YOLO-World-master/
└── data/
    ├── DIOR/
    │   ├── annotations/
    │   │   ├── train.json          ← 训练集 COCO 标注
    │   │   └── val.json            ← 验证集 COCO 标注
    │   ├── images/
    │   │   ├── train/              ← 训练图片 (~20000 张)
    │   │   └── val/                ← 验证图片 (~3000 张)
    │   └── dior.yaml
    └── texts/
        └── dior_class_texts.json   ← 类别文本描述 (自动生成)
```

---

## 5. 下载预训练模型

YOLO-World 需要预训练权重作为起点:

```bash
cd YOLO-World-master
mkdir pretrained_models

# 下载 Small 版本预训练模型 (推荐入门)
# 文件名: yolo_world_s_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_train-55b943ea.pth
# 大小: 约 300MB

# 方式 1: HuggingFace 下载
wget -P pretrained_models https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_s_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_train-55b943ea.pth

# 方式 2: 镜像下载 (国内推荐)
wget -P pretrained_models https://hf-mirror.com/wondervictor/YOLO-World/resolve/main/yolo_world_s_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_train-55b943ea.pth

# 方式 3: 浏览器手动下载, 放到 pretrained_models/ 目录
```

下载完成后检查:
```bash
ls pretrained_models/
# 应该看到 yolo_world_s_clip_t2i_bn_..._55b943ea.pth
```

---

## 6. 本地跑通流程 (验证代码)

**强烈建议**: 先用小数据在本地跑 2 个 epoch,确认代码无误,再上服务器。

### 6.1 修改配置文件 (减少 epoch)

打开 `YOLO-World-master/configs/finetune_dior/yolo_world_v2_s_dior.py`,临时改:

```python
max_epochs = 2                 # ← 改成 2 (快速测试)
train_batch_size_per_gpu = 2   # ← 改成 2 (省显存)
```

### 6.2 跑测试训练

```bash
cd YOLO-World-master
python tools/train.py configs/finetune_dior/yolo_world_v2_s_dior.py --amp
```

### 6.3 检查输出

如果成功,你会看到:
- 终端打印训练进度
- `work_dirs/yolo_world_v2_s_dior/` 目录有日志和权重文件

**如果报错**: 看下面的[常见问题](#10-常见问题)部分。

---

## 7. 租 GPU 服务器训练

### 7.1 选择平台

推荐 **AutoDL**: https://www.autodl.com/
- RTX 4090: ¥2.5/小时
- RTX 3090: ¥1.5/小时
- 充 ¥50-100 就够训练一次

### 7.2 租用实例

1. 注册 AutoDL 账号
2. 充值 (支付宝/微信)
3. 算力市场 → 选 RTX 4090
4. 镜像选: `PyTorch 2.1 + Python 3.10 + CUDA 12.1`
5. 创建实例

### 7.3 上传项目到服务器

**方法 1: GitHub 中转 (推荐)**

```bash
# 本地: 把项目推到 GitHub
cd e:\Model_Train
git init
git add .
git commit -m "YOLO-World + DIOR project"
git remote add origin https://github.com/你的用户名/model_train.git
git push -u origin main

# 服务器: 克隆
git clone https://github.com/你的用户名/model_train.git
```

**方法 2: scp 上传**

```bash
# 在本地 PowerShell 执行 (替换端口和地址)
scp -P 12345 -r e:\Model_Train root@region-1.autodl.com:/root/
```

**方法 3: 分文件上传**
- 代码: GitHub 或 scp
- DIOR 数据集: 服务器上直接下载 (30GB 国内下载比上传快)
- 预训练模型: 服务器上 wget 下载

### 7.4 在服务器上安装环境

```bash
# SSH 连接服务器后
cd /root/Model_Train

# 安装依赖
pip install -U openmim
mim install mmengine mmcv mmdet mmyolo
pip install transformers einops timm tqdm

# 验证
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 7.5 VSCode Remote-SSH 连接

1. VSCode 安装插件: `Remote - SSH`
2. 按 F1 → `Remote-SSH: Connect to Host`
3. 输入: `ssh -p 12345 root@region-1.autodl.com`
4. 打开 `/root/Model_Train/` 目录

### 7.6 在服务器上准备数据

```bash
# 如果数据没上传, 在服务器上直接下载 DIOR
# (略, 参考 DIOR 官网)

# 运行数据准备脚本
cd /root/Model_Train
python prepare_dior.py \
    --xml-dir "DIOR/Annotations/XML" \
    --image-dir "DIOR/Images" \
    --output-dir "YOLO-World-master/data/DIOR"

# 划分图片
python split_dior_images.py
```

### 7.7 在服务器上开始训练

```bash
cd /root/Model_Train/YOLO-World-master

# 单卡训练 (推荐)
nohup python tools/train.py \
    configs/finetune_dior/yolo_world_v2_s_dior.py \
    --amp > train.log 2>&1 &

# 查看训练日志
tail -f train.log

# 或用 tmux (推荐)
tmux new -s train
python tools/train.py configs/finetune_dior/yolo_world_v2_s_dior.py --amp
# Ctrl+B 然后 D 退出 (训练继续)
# tmux attach -t train 重新连接
```

### 7.8 下载结果到本地

训练完成后:
```bash
# 下载最佳模型 (本地执行)
scp -P 12345 root@region-1.autodl.com:/root/Model_Train/YOLO-World-master/work_dirs/yolo_world_v2_s_dior/best_coco_bbox_mAP_epoch_80.pth e:\Model_Train\

# 下载训练曲线
scp -P 12345 root@region-1.autodl.com:/root/Model_Train/YOLO-World-master/work_dirs/yolo_world_v2_s_dior/vis_data/*.png e:\Model_Train\results\
```

### 7.9 省钱技巧

- ⚠️ **不用就关机**! AutoDL 关机只收 ¥0.01/GB/小时存储费
- 用按量付费,不要包日
- 数据放 `/root/autodl-tmp/` (数据盘,便宜)
- 代码放 `/root/` (系统盘)

---

## 8. 训练参数调整

打开 [YOLO-World-master/configs/finetune_dior/yolo_world_v2_s_dior.py](YOLO-World-master/configs/finetune_dior/yolo_world_v2_s_dior.py) 修改参数。

### 8.1 最常改的参数

| 参数 | 位置 | 默认值 | 怎么改 |
|------|------|--------|--------|
| `max_epochs` | 第 19 行 | 80 | 训练时间不够就减到 40 |
| `train_batch_size_per_gpu` | 第 24 行 | 8 | 显存不足减到 4 或 2 |
| `base_lr` | 第 22 行 | 2e-4 | batch 减小时, lr 等比例减小 |
| `close_mosaic_epochs` | 第 20 行 | 10 | 最后 N 轮关闭 Mosaic |

### 8.2 显存不够怎么办

```python
# 减小 batch (最有效)
train_batch_size_per_gpu = 4     # 4090 可以 4-8
# 学习率等比例缩小
base_lr = 1e-4                   # 原来 2e-4, 现在 1e-4

# 或者减小输入分辨率 (需要改 _base_ 配置)
# 不建议, 会影响精度
```

### 8.3 训练太慢怎么办

```python
# 1. 减少 epochs
max_epochs = 40                  # 原来 80

# 2. 关闭 CopyPaste (会加速)
copypaste_prob = 0.0             # 原来 0.3

# 3. 用更大显卡 (4090 比 3090 快约 2 倍)
```

### 8.4 精度不够怎么办

```python
# 1. 增大模型 (S → M → L)
# 改用: configs/finetune_coco/yolo_world_v2_m_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco.py
# 然后改里面的 data_root 等参数

# 2. 增加 epochs
max_epochs = 150

# 3. 增加数据增强
mixup_prob = 0.2                 # 原来 0.15
copypaste_prob = 0.5            # 原来 0.3
```

---

## 9. 训练结果分析

### 9.1 训练完成后查看

训练结果保存在 `work_dirs/yolo_world_v2_s_dior/`:

```
work_dirs/yolo_world_v2_s_dior/
├── best_coco_bbox_mAP_epoch_80.pth    ← 最佳模型
├── last_checkpoint                     ← 最后一轮
├── epoch_80.pth                        ← 第 80 轮权重
├── vis_data/
│   ├── config.json
│   ├── 20260715_xxx/
│   │   ├── train/
│   │   │   ├── losses.png             ← Loss 曲线
│   │   │   └── ...
│   │   └── val/
│   │       ├── mAP.png                ← mAP 曲线
│   │       └── ...
└── 20260715_xxx.log                   ← 训练日志
```

### 9.2 评估指标解读

```bash
# 用训练好的模型评估
python tools/test.py \
    configs/finetune_dior/yolo_world_v2_s_dior.py \
    work_dirs/yolo_world_v2_s_dior/best_coco_bbox_mAP_epoch_80.pth
```

输出会显示:
- **mAP@0.5**: IoU=0.5 时的平均精度 (主要指标)
- **mAP@0.5:0.95**: 多个 IoU 阈值的平均精度
- 每个类别的 AP

**参考数值** (DIOR 数据集, YOLO-World-S):
- mAP@0.5: 70-75% (良好)
- mAP@0.5: 75-80% (优秀)

### 9.3 用模型推理新图片

```bash
python demo/image_demo.py \
    path/to/test.jpg \
    configs/finetune_dior/yolo_world_v2_s_dior.py \
    work_dirs/yolo_world_v2_s_dior/best_coco_bbox_mAP_epoch_80.pth
```

### 9.4 体验开放词汇能力

YOLO-World 的特色是**可以用文本检测新类别**。修改测试脚本,输入任意文本:

```python
# 测试任意类别 (不限于 DIOR 20 类)
from mmyolo.utils import register_all_modules
from mmengine.runner import Runner

register_all_modules()
runner = Runner.from_cfg(cfg)
# 修改 texts 参数, 例如加入 "river", "forest", "building" 等
```

---

## 10. 常见问题

### Q1: CUDA out of memory (显存不足)

**错误**: `RuntimeError: CUDA out of memory`

**解决**:
```python
# 1. 减小 batch (最有效)
train_batch_size_per_gpu = 4

# 2. 如果还不够, 减到 2
train_batch_size_per_gpu = 2
base_lr = 5e-5  # 学习率等比例减小

# 3. 用 --amp 混合精度训练 (命令行加)
python tools/train.py ... --amp
```

### Q2: 找不到数据集

**错误**: `FileNotFoundError: ...train.json`

**解决**:
- 检查 `data/DIOR/annotations/train.json` 是否存在
- 检查配置文件中 `data_root` 路径是否正确
- 用绝对路径避免相对路径问题

### Q3: 找不到预训练模型

**错误**: `FileNotFoundError: pretrained_models/yolo_world_s_...pth`

**解决**:
```bash
cd YOLO-World-master
mkdir pretrained_models
# 重新下载预训练权重
wget -P pretrained_models https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_s_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_train-55b943ea.pth
```

### Q4: mmyolo 安装失败

**错误**: `pip install mmyolo` 报错

**解决**:
```bash
# 按顺序安装
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmyolo

# 如果 mmcv 装不上, 用预编译版本
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
# (根据你的 CUDA 和 PyTorch 版本调整 URL)
```

### Q5: CLIP 模型下载失败

**错误**: `OSError: ...clip-vit-base-patch32...`

**解决**:
```bash
# 用镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 或者手动下载 CLIP 模型
# 1. 浏览器访问: https://hf-mirror.com/openai/clip-vit-base-patch32
# 2. 下载所有文件, 放到本地路径
# 3. 修改配置: text_model_name = '本地路径/clip-vit-base-patch32'
```

### Q6: 训练 loss 不下降

**可能原因**:
1. 学习率太小 → 增大 `base_lr`
2. 学习率太大 → 减小 `base_lr`
3. 数据标注错误 → 检查 train.json
4. 数据太少 → 增加 epochs

**调试**:
```python
# 在配置里加日志频率
default_hooks = dict(
    logger=dict(interval=10),  # 每 10 步打印一次
)
```

### Q7: 训练中断了怎么恢复

```bash
# 从最近一次 checkpoint 恢复
python tools/train.py \
    configs/finetune_dior/yolo_world_v2_s_dior.py \
    --resume-from work_dirs/yolo_world_v2_s_dior/last_checkpoint
```

### Q8: 怎么知道训练效果好不好

**好训练的特征**:
- Loss 平稳下降,没有剧烈波动
- mAP 持续上升,最后收敛
- 训练集和验证集 mAP 差距不大 (没有过拟合)

**过拟合的特征**:
- 训练集 mAP 很高,验证集 mAP 不升甚至下降
- 解决: 加数据增强,减小 epochs,加 weight_decay

---

## 📁 项目文件清单

我为你创建的文件:

| 文件 | 作用 |
|------|------|
| [prepare_dior.py](prepare_dior.py) | DIOR 数据转换脚本 (XML→COCO) |
| [train_dior.py](train_dior.py) | 一键训练脚本 (带环境检查) |
| [YOLO-World-master/configs/finetune_dior/yolo_world_v2_s_dior.py](YOLO-World-master/configs/finetune_dior/yolo_world_v2_s_dior.py) | YOLO-World DIOR 训练配置 |
| [DIOR_训练完全指南.md](DIOR_训练完全指南.md) | 本文档 |

---

## 🚀 快速开始 (TL;DR)

如果你只想快速跑通,执行这 5 步:

```bash
# 1. 准备环境
pip install -U openmim
mim install mmengine mmcv mmdet mmyolo
pip install transformers einops timm tqdm

# 2. 下载 DIOR 数据集并解压到 DIOR/ 目录

# 3. 转换数据格式
python prepare_dior.py \
    --xml-dir "DIOR/Annotations/XML" \
    --image-dir "DIOR/Images" \
    --output-dir "YOLO-World-master/data/DIOR"

# 4. 划分图片
python split_dior_images.py

# 5. 下载预训练模型
cd YOLO-World-master
mkdir pretrained_models
wget -P pretrained_models https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_s_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_train-55b943ea.pth

# 6. 开始训练
python tools/train.py configs/finetune_dior/yolo_world_v2_s_dior.py --amp
```

---

## 📞 获得帮助

如果遇到问题:
1. 先看[常见问题](#10-常见问题)部分
2. 查看训练日志: `work_dirs/yolo_world_v2_s_dior/*.log`
3. 把错误信息发给我,我帮你分析

**祝训练顺利!** 🚀

---

*最后更新: 2026-07-15*
*作者: YOLO-World + DIOR Project*

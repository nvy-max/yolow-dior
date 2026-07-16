#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIOR 图片划分脚本
把 DIOR/Images/ 下的图片按 train.json / val.json 划分到 train/ 和 val/ 目录
"""

import json
import os
import shutil
import sys
from pathlib import Path


def split_images(json_file, src_img_dir, dst_img_dir):
    """根据 JSON 标注, 把对应图片复制到目标目录"""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(dst_img_dir, exist_ok=True)

    n_copied = 0
    n_missing = 0

    for img in data["images"]:
        filename = img["file_name"]
        # 处理带子目录的路径 (如 "train/1.jpg")
        if "/" in filename:
            filename = os.path.basename(filename)

        src = os.path.join(src_img_dir, filename)
        dst = os.path.join(dst_img_dir, filename)

        if os.path.exists(src):
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                n_copied += 1
        else:
            n_missing += 1
            if n_missing <= 3:
                print(f"  [警告] 图片不存在: {src}")

    print(f"  复制 {n_copied} 张图片到 {dst_img_dir}")
    if n_missing > 0:
        print(f"  [警告] {n_missing} 张图片缺失")

    return n_copied


def main():
    # 默认路径 (可根据参数修改)
    base = "YOLO-World-master/data/DIOR"
    src = "DIOR/Images"

    if len(sys.argv) >= 4:
        base = sys.argv[1]
        src = sys.argv[2]
    elif len(sys.argv) >= 2:
        base = sys.argv[1]

    train_json = os.path.join(base, "annotations", "train.json")
    val_json = os.path.join(base, "annotations", "val.json")

    print("=" * 50)
    print("  DIOR 图片划分")
    print("=" * 50)
    print(f"  数据目录: {base}")
    print(f"  源图片: {src}")
    print()

    # 检查 JSON 是否存在
    for j in [train_json, val_json]:
        if not os.path.exists(j):
            print(f"  [错误] 找不到 {j}")
            print(f"         请先运行 prepare_dior.py")
            sys.exit(1)

    # 检查源图片目录
    if not os.path.isdir(src):
        print(f"  [错误] 源图片目录不存在: {src}")
        sys.exit(1)

    # 划分训练集
    print("[1/2] 划分训练图片...")
    n1 = split_images(train_json, src, os.path.join(base, "images", "train"))

    # 划分验证集
    print("\n[2/2] 划分验证图片...")
    n2 = split_images(val_json, src, os.path.join(base, "images", "val"))

    # 完成
    print("\n" + "=" * 50)
    print(f"  完成! 训练 {n1} 张, 验证 {n2} 张")
    print("=" * 50)
    print(f"""
  目录结构:
    {base}/
    ├── annotations/
    │   ├── train.json
    │   └── val.json
    └── images/
        ├── train/   ({n1} 张)
        └── val/     ({n2} 张)

  下一步: 开始训练
    cd YOLO-World-master
    python tools/train.py configs/finetune_dior/yolo_world_v2_s_dior.py --amp
""")


if __name__ == "__main__":
    main()

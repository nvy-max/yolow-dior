#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================
DIOR 数据集准备脚本 (小白专用)
=============================================================
功能:
    1. 把 DIOR 的 XML 标注转成 YOLO-World 需要的 COCO JSON 格式
    2. 生成 YOLO-World 开放词汇训练需要的类别文本描述文件
    3. 自动创建训练/验证集目录结构

使用方法:
    python prepare_dior.py --xml-dir 数据集/Annotations/XML \
                          --image-dir 数据集/Images \
                          --output-dir data/DIOR

数据集下载:
    DIOR 官网: http://www.uestc.edu.cn/DIOR.html
    Google Drive: 搜索 "DIOR dataset download"
=============================================================
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm


# ============================================================
# DIOR 数据集 20 个类别 (顺序固定,不要改!)
# ============================================================
DIOR_CLASSES = [
    "airplane",                  # 0 飞机
    "airport",                    # 1 机场
    "baseball-diamond",          # 2 棒球场
    "basketball-court",          # 3 篮球场
    "bridge",                     # 4 桥
    "chimney",                    # 5 烟囱
    "expressway-service-area",   # 6 高速服务区
    "expressway-toll-station",    # 7 高速收费站
    "dam",                        # 8 大坝
    "golf-course",               # 9 高尔夫球场
    "ground-track-field",        # 10 田径场
    "harbor",                     # 11 港口
    "overpass",                   # 12 立交桥
    "ship",                       # 13 船
    "stadium",                    # 14 体育场
    "storage-tank",              # 15 储罐
    "tennis-court",              # 16 网球场
    "train-station",             # 17 火车站
    "vehicle",                    # 18 车辆
    "windmill"                    # 19 风车
]


# ============================================================
# YOLO-World 需要的类别文本描述 (每个类多个描述,提升泛化能力)
# ============================================================
DIOR_CLASS_TEXTS = [
    ["airplane", "a flying airplane", "aircraft", "passenger plane"],
    ["airport", "airport with runway", "aviation airport", "aerodrome"],
    ["baseball-diamond", "baseball diamond", "baseball field"],
    ["basketball-court", "basketball court", "basketball field"],
    ["bridge", "a bridge over river", "road bridge", "arch bridge"],
    ["chimney", "industrial chimney", "smokestack", "factory chimney"],
    ["expressway-service-area", "expressway service area", "highway rest area"],
    ["expressway-toll-station", "expressway toll station", "toll booth"],
    ["dam", "water dam", "concrete dam", "hydroelectric dam"],
    ["golf-course", "golf course", "golf field", "golf links"],
    ["ground-track-field", "ground track field", "running track", "athletics track"],
    ["harbor", "harbor", "port", "docking harbor with ships"],
    ["overpass", "overpass", "highway overpass", "flyover"],
    ["ship", "ship", "vessel", "boat in water", "cargo ship"],
    ["stadium", "stadium", "sports stadium", "arena"],
    ["storage-tank", "storage tank", "oil tank", "industrial storage tank"],
    ["tennis-court", "tennis court", "tennis field"],
    ["train-station", "train station", "railway station"],
    ["vehicle", "vehicle", "car", "truck", "automobile"],
    ["windmill", "windmill", "wind turbine", "wind power generator"]
]


def parse_args():
    p = argparse.ArgumentParser(description="DIOR 数据集准备 (小白专用)")
    p.add_argument("--xml-dir", type=str, required=True,
                   help="DIOR 标注 XML 目录 (如: 数据集/Annotations/XML)")
    p.add_argument("--image-dir", type=str, required=True,
                   help="DIOR 图片目录 (如: 数据集/Images)")
    p.add_argument("--output-dir", type=str, default="data/DIOR",
                   help="输出目录 (默认: data/DIOR)")
    p.add_argument("--train-ratio", type=float, default=0.85,
                   help="训练集比例 (默认 0.85, 剩下做验证)")
    return p.parse_args()


def voc_xml_to_coco(xml_files, image_dir, output_json, image_prefix=""):
    """把 DIOR XML 标注转成 COCO JSON 格式"""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": c, "supercategory": "dior"}
            for i, c in enumerate(DIOR_CLASSES)
        ]
    }

    ann_id = 1
    img_id = 1

    print(f"  正在转换 {len(xml_files)} 个 XML 文件...")

    for xml_file in tqdm(xml_files, desc="  转换进度"):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(image_dir.replace("Images", ""), "Annotations", "XML", xml_file) \
            if "Annotations" not in image_dir else os.path.join(os.path.dirname(os.path.dirname(image_dir)),
                                                                 "Annotations", "XML", xml_file)

        # 如果上面路径不对,直接用传入的 xml_dir
        if not os.path.exists(xml_path):
            xml_path = xml_file  # 假设是完整路径

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"  [跳过] 无法解析 {xml_file}: {e}")
            continue

        # 图片信息
        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        coco["images"].append({
            "id": img_id,
            "file_name": os.path.join(image_prefix, filename) if image_prefix else filename,
            "width": width,
            "height": height
        })

        # 标注框
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in DIOR_CLASSES:
                continue
            cat_id = DIOR_CLASSES.index(name) + 1

            bbox = obj.find("bndbox")
            # DIOR: xmin, ymin, xmax, ymax
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)

            # xyxy → xywh (COCO 格式)
            w = x2 - x1
            h = y2 - y1

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    # 保存
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"  ✅ 转换完成: {output_json}")
    print(f"     图片数: {img_id - 1}")
    print(f"     标注数: {ann_id - 1}")
    return coco


def create_class_texts(output_path):
    """生成 YOLO-World 需要的类别文本描述 JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(DIOR_CLASS_TEXTS, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 类别文本描述已生成: {output_path}")
    print(f"     类别数: {len(DIOR_CLASS_TEXTS)}")


def main():
    args = parse_args()

    print("=" * 60)
    print("  DIOR 数据集准备工具 (YOLO-World 专用)")
    print("=" * 60)
    print(f"  XML 目录: {args.xml_dir}")
    print(f"  图片目录: {args.image_dir}")
    print(f"  输出目录: {args.output_dir}")
    print()

    # 1. 检查输入
    if not os.path.isdir(args.xml_dir):
        print(f"  ❌ XML 目录不存在: {args.xml_dir}")
        return
    if not os.path.isdir(args.image_dir):
        print(f"  ❌ 图片目录不存在: {args.image_dir}")
        return

    # 2. 获取所有 XML 文件
    xml_files = sorted([f for f in os.listdir(args.xml_dir) if f.endswith(".xml")])
    print(f"  发现 {len(xml_files)} 个 XML 标注文件")

    if len(xml_files) == 0:
        print("  ❌ 没有找到 XML 文件,请检查路径")
        return

    # 3. 划分训练/验证集
    import random
    random.seed(42)
    random.shuffle(xml_files)

    n_train = int(len(xml_files) * args.train_ratio)
    train_xmls = xml_files[:n_train]
    val_xmls = xml_files[n_train:]

    print(f"  训练集: {len(train_xmls)} 张")
    print(f"  验证集: {len(val_xmls)} 张")
    print()

    # 4. 转换为 COCO 格式
    print("[步骤 1/3] 转换训练集...")
    voc_xml_to_coco(
        xml_files=[os.path.join(args.xml_dir, f) for f in train_xmls],
        image_dir=args.image_dir,
        output_json=os.path.join(args.output_dir, "annotations", "train.json"),
        image_prefix="train"
    )

    print("\n[步骤 2/3] 转换验证集...")
    voc_xml_to_coco(
        xml_files=[os.path.join(args.xml_dir, f) for f in val_xmls],
        image_dir=args.image_dir,
        output_json=os.path.join(args.output_dir, "annotations", "val.json"),
        image_prefix="val"
    )

    # 5. 生成类别文本
    print("\n[步骤 3/3] 生成类别文本描述...")
    create_class_texts(os.path.join("data", "texts", "dior_class_texts.json"))

    # 6. 创建数据集 yaml 配置文件
    yaml_content = f"""# DIOR 数据集配置 (自动生成)
# 用于 YOLO-World 训练

# 数据集路径
path: {args.output_dir}

# 训练集和验证集
train: annotations/train.json
val: annotations/val.json

# 类别数
nc: 20

# 类别名称
names:
  0: airplane
  1: airport
  2: baseball-diamond
  3: basketball-court
  4: bridge
  5: chimney
  6: expressway-service-area
  7: expressway-toll-station
  8: dam
  9: golf-course
  10: ground-track-field
  11: harbor
  12: overpass
  13: ship
  14: stadium
  15: storage-tank
  16: tennis-court
  17: train-station
  18: vehicle
  19: windmill
"""
    yaml_path = os.path.join(args.output_dir, "dior.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\n  ✅ 数据集配置: {yaml_path}")

    # 7. 完成提示
    print("\n" + "=" * 60)
    print("  数据准备完成!")
    print("=" * 60)
    print(f"""
  下一步:
    1. 把图片按 train/val 划分到对应目录:
       {args.output_dir}/images/train/
       {args.output_dir}/images/val/

    2. 确认目录结构:
       {args.output_dir}/
       ├── annotations/
       │   ├── train.json
       │   └── val.json
       ├── images/
       │   ├── train/   ← 训练图片
       │   └── val/     ← 验证图片
       └── dior.yaml

    3. 开始训练:
       cd YOLO-World-master
       python tools/train.py configs/finetune_dior/yolo_world_v2_s_dior.py --amp
""")


if __name__ == "__main__":
    main()

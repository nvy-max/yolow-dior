#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================
YOLO-World + DIOR 一键训练脚本 (小白专用)
=============================================================
功能:
    1. 自动检查环境 (PyTorch, mmyolo, GPU)
    2. 自动检查数据集是否准备好
    3. 自动检查预训练模型是否存在
    4. 一键启动训练
    5. 训练后自动评估

使用方法:
    python train_dior.py                       # 单卡训练
    python train_dior.py --gpus 4              # 4 卡训练
    python train_dior.py --epochs 40           # 改 epochs
    python train_dior.py --batch 4             # 显存不够改小 batch
=============================================================
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Windows 中文乱码修复
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ============================================================
# 默认配置 (和配置文件保持一致)
# ============================================================
DEFAULT_CONFIG = {
    # 训练配置
    "config_file": "configs/finetune_dior/yolo_world_v2_s_dior.py",
    "epochs": 80,
    "batch_size": 8,
    "gpus": 1,

    # 路径
    "project_root": "YOLO-World-master",
    "data_root": "data/DIOR",
    "pretrained_dir": "pretrained_models",
    "pretrained_model": "yolo_world_s_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_train-55b943ea.pth",
    "pretrained_url": "https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_s_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_train-55b943ea.pth",

    # CLIP 模型 (会自动下载)
    "clip_model": "openai/clip-vit-base-patch32",
}


def print_banner():
    """打印启动 banner"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   YOLO-World + DIOR 数据集训练脚本                        ║
║   Real-Time Open-Vocabulary Object Detection             ║
║                                                           ║
║   数据集: DIOR (20 类遥感目标)                            ║
║   模型:   YOLO-World v2-S                                 ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)


def parse_args():
    p = argparse.ArgumentParser(description="YOLO-World + DIOR 训练")
    p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
                   help=f"训练轮数 (默认 {DEFAULT_CONFIG['epochs']})")
    p.add_argument("--batch", type=int, default=DEFAULT_CONFIG["batch_size"],
                   help=f"每卡 batch size (默认 {DEFAULT_CONFIG['batch_size']})")
    p.add_argument("--gpus", type=int, default=DEFAULT_CONFIG["gpus"],
                   help=f"GPU 数量 (默认 {DEFAULT_CONFIG['gpus']})")
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG["config_file"],
                   help="训练配置文件路径")
    p.add_argument("--no-amp", action="store_true",
                   help="禁用混合精度 (省显存但慢)")
    p.add_argument("--dry-run", action="store_true",
                   help="只检查环境, 不真正训练")
    return p.parse_args()


def check_python_version():
    """检查 Python 版本"""
    print("[1/5] 检查 Python 版本...")
    version = sys.version_info
    print(f"      Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("      ❌ 需要 Python 3.8+, 请升级")
        return False
    print("      ✅ Python 版本符合要求")
    return True


def check_torch():
    """检查 PyTorch 和 GPU"""
    print("\n[2/5] 检查 PyTorch 和 GPU...")
    try:
        import torch
        print(f"      PyTorch 版本: {torch.__version__}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"      ✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            print(f"      CUDA 版本: {torch.version.cuda}")
            return True
        else:
            print("      ⚠️  未检测到 GPU!")
            print("      训练会非常慢, 建议租 GPU 服务器")
            ans = input("      仍然继续? (y/n): ").lower().strip()
            return ans == "y"
    except ImportError:
        print("      ❌ 未安装 PyTorch")
        print("      安装: pip install torch torchvision")
        return False


def check_mmyolo():
    """检查 mmyolo 全家桶"""
    print("\n[3/5] 检查 mmyolo 环境...")
    required = ["mmcv", "mmdet", "mmyolo", "mmengine"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
            print(f"      ✅ {pkg}")
        except ImportError:
            print(f"      ❌ {pkg} 未安装")
            missing.append(pkg)

    if missing:
        print(f"\n      缺少依赖: {', '.join(missing)}")
        print("      一键安装命令:")
        print("        pip install -U openmim")
        print("        mim install mmengine mmcv mmdet mmyolo")
        ans = input("\n      是否现在自动安装? (y/n): ").lower().strip()
        if ans == "y":
            os.system("pip install -U openmim")
            os.system("mim install mmengine mmcv mmdet mmyolo")
            # 重新检查
            for pkg in missing:
                try:
                    __import__(pkg)
                    print(f"      ✅ {pkg} 安装成功")
                except ImportError:
                    print(f"      ❌ {pkg} 安装失败, 请手动安装")
                    return False
        else:
            return False

    print("      ✅ mmyolo 环境完整")
    return True


def check_dataset():
    """检查 DIOR 数据集"""
    print("\n[4/5] 检查 DIOR 数据集...")
    data_root = DEFAULT_CONFIG["data_root"]

    # 检查标注文件
    train_json = os.path.join(data_root, "annotations", "train.json")
    val_json = os.path.join(data_root, "annotations", "val.json")
    texts_json = "data/texts/dior_class_texts.json"

    for name, path in [("训练标注", train_json),
                        ("验证标注", val_json),
                        ("类别文本", texts_json)]:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"      ✅ {name}: {path} ({size:.1f} KB)")
        else:
            print(f"      ❌ {name} 不存在: {path}")
            print(f"         请先运行: python prepare_dior.py")
            return False

    # 检查图片目录
    train_img_dir = os.path.join(data_root, "images", "train")
    val_img_dir = os.path.join(data_root, "images", "val")

    for name, path in [("训练图片", train_img_dir), ("验证图片", val_img_dir)]:
        if os.path.isdir(path):
            n = len(os.listdir(path))
            print(f"      ✅ {name}: {path} ({n} 个文件)")
            if n == 0:
                print(f"      ❌ {name} 目录为空")
                return False
        else:
            print(f"      ❌ {name} 目录不存在: {path}")
            return False

    print("      ✅ 数据集完整")
    return True


def check_pretrained():
    """检查预训练模型"""
    print("\n[5/5] 检查预训练模型...")
    pretrained_dir = DEFAULT_CONFIG["pretrained_dir"]
    model_name = DEFAULT_CONFIG["pretrained_model"]
    model_path = os.path.join(pretrained_dir, model_name)

    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / 1024 / 1024
        print(f"      ✅ 预训练模型: {model_path} ({size:.1f} MB)")
        return True
    else:
        print(f"      ❌ 预训练模型不存在: {model_path}")
        print(f"      下载地址: {DEFAULT_CONFIG['pretrained_url']}")
        ans = input("      是否现在自动下载? (y/n): ").lower().strip()
        if ans == "y":
            os.makedirs(pretrained_dir, exist_ok=True)
            print(f"      下载中... (约 300MB, 可能需要几分钟)")
            url = DEFAULT_CONFIG["pretrained_url"]
            cmd = f"wget -P {pretrained_dir} {url}"
            os.system(cmd)
            if os.path.exists(model_path):
                print(f"      ✅ 下载完成")
                return True
            else:
                print(f"      ❌ 下载失败, 请手动下载")
                print(f"         下载后放到: {model_path}")
                return False
        return False


def run_training(args):
    """执行训练"""
    print("\n" + "=" * 60)
    print("  开始训练")
    print("=" * 60)

    # 进入 YOLO-World 项目目录
    project_root = DEFAULT_CONFIG["project_root"]
    if not os.path.isdir(project_root):
        print(f"  ❌ YOLO-World 目录不存在: {project_root}")
        return

    os.chdir(project_root)
    print(f"  工作目录: {os.getcwd()}")
    print(f"  配置文件: {args.config}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  每卡 batch: {args.batch}")
    print(f"  GPU 数量: {args.gpus}")

    # 构造训练命令
    if args.gpus == 1:
        # 单卡训练
        cmd = [
            "python", "tools/train.py",
            args.config,
            "--amp" if not args.no_amp else "",
        ]
        # 移除空字符串
        cmd = [c for c in cmd if c]
    else:
        # 多卡训练
        cmd = [
            f"./tools/dist_train.sh",
            args.config,
            str(args.gpus),
        ]
        if not args.no_amp:
            cmd.append("--amp")

    print(f"\n  训练命令: {' '.join(cmd)}")
    print(f"\n  训练过程中:")
    print(f"    - 实时日志在终端显示")
    print(f"    - 权重保存在 work_dirs/ 目录")
    print(f"    - 按 Ctrl+C 可中断训练 (断点可恢复)")
    print(f"\n  {'=' * 60}")

    if args.dry_run:
        print("\n  [dry-run] 跳过实际训练")
        return

    # 执行训练
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n  ❌ 训练失败: {e}")
        print(f"     常见问题:")
        print(f"     - CUDA OOM: 减小 batch (--batch 4 或 2)")
        print(f"     - 数据找不到: 检查 data/DIOR/ 路径")
        print(f"     - 模型找不到: 检查 pretrained_models/ 目录")
        return
    except KeyboardInterrupt:
        print(f"\n\n  ⚠️  训练被中断")
        print(f"     可用 --resume-from 恢复训练")
        return

    # 训练完成
    print("\n" + "=" * 60)
    print("  🎉 训练完成!")
    print("=" * 60)
    print(f"""
  结果位置:
    - 最佳权重: work_dirs/yolo_world_v2_s_dior/best_coco_bbox_mAP_epoch_*.pth
    - 最末权重: work_dirs/yolo_world_v2_s_dior/last_checkpoint
    - 训练日志: work_dirs/yolo_world_v2_s_dior/*.log
    - 训练曲线: work_dirs/yolo_world_v2_s_dior/vis_data/*.png

  下一步:
    1. 查看训练曲线, 确认 loss 下降, mAP 上升
    2. 用训练好的模型做推理测试:
       python demo/image_demo.py path/to/image.jpg \\
              configs/finetune_dior/yolo_world_v2_s_dior.py \\
              work_dirs/yolo_world_v2_s_dior/best_coco_bbox_mAP_epoch_80.pth

    3. 导出 ONNX 部署:
       python deploy/export_onnx.py \\
              configs/finetune_dior/yolo_world_v2_s_dior.py \\
              work_dirs/yolo_world_v2_s_dior/best_coco_bbox_mAP_epoch_80.pth
""")


def main():
    args = parse_args()

    print_banner()

    # 1. 环境检查
    print("=" * 60)
    print("  环境检查")
    print("=" * 60)

    if not check_python_version():
        sys.exit(1)

    if not check_torch():
        sys.exit(1)

    if not check_mmyolo():
        sys.exit(1)

    if not check_dataset():
        print("\n  ❌ 数据集检查失败")
        print("     请先运行: python prepare_dior.py")
        sys.exit(1)

    if not check_pretrained():
        sys.exit(1)

    # 2. 开始训练
    run_training(args)


if __name__ == "__main__":
    main()

#!/bin/bash
# ============================================================
# AutoDL 服务器环境一键配置脚本 (开机后第一个跑的脚本)
# ============================================================
# 功能:
#   1. 安装 mmyolo 全家桶
#   2. 安装其他依赖
#   3. 配置 huggingface 镜像 (国内加速)
#   4. 克隆你的项目 (改成你的 GitHub 地址)
#   5. 下载预训练模型
#
# 使用方法:
#   bash setup_autodl.sh
# ============================================================

set -e  # 遇到错误立即停止

echo "============================================================"
echo "  AutoDL 环境一键配置 (YOLO-World + DIOR)"
echo "============================================================"

# 1. 检查 GPU
echo ""
echo "[1/6] 检查 GPU..."
nvidia-smi
echo ""

# 2. 配置 pip 镜像 (国内加速)
echo "[2/6] 配置 pip 镜像..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 3. 配置 huggingface 镜像 (下载 CLIP 模型用)
echo "[3/6] 配置 huggingface 镜像..."
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# 4. 安装 mmyolo 全家桶
echo "[4/6] 安装 mmyolo..."
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmyolo

# 5. 安装其他依赖
echo "[5/6] 安装其他依赖..."
pip install transformers einops timm tqdm opencv-python matplotlib

# 6. 验证安装
echo "[6/6] 验证安装..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}, CUDA: {torch.version.cuda}')"
python -c "import mmyolo; print('mmyolo: OK')"
python -c "import mmcv; print(f'mmcv: {mmcv.__version__}')"
python -c "import mmdet; print(f'mmdet: {mmdet.__version__}')"

echo ""
echo "============================================================"
echo "  环境配置完成!"
echo "============================================================"
echo ""
echo "下一步:"
echo "  1. 上传项目代码:"
echo "     git clone https://github.com/你的用户名/yolow-dior.git"
echo ""
echo "  2. 下载 DIOR 数据集 (30GB, 可能需要 10-30 分钟):"
echo "     # 把下载命令写在这里 (根据你的下载来源)"
echo ""
echo "  3. 下载预训练模型:"
echo "     cd yolow-dior/YOLO-World-master"
echo "     mkdir -p pretrained_models"
echo "     wget -P pretrained_models https://hf-mirror.com/wondervictor/YOLO-World/resolve/main/yolo_world_s_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_train-55b943ea.pth"
echo ""
echo "  4. 准备数据:"
echo "     cd yolow-dior"
echo "     python prepare_dior.py --xml-dir DIOR/Annotations/XML --image-dir DIOR/Images --output-dir YOLO-World-master/data/DIOR"
echo "     python split_dior_images.py"
echo ""
echo "  5. 开始训练:"
echo "     cd YOLO-World-master"
echo "     tmux new -s train"
echo "     python tools/train.py configs/finetune_dior/yolo_world_v2_s_dior.py --amp"
echo "     # Ctrl+B 然后 D 退出 tmux"
echo ""

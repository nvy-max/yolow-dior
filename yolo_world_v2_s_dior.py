# ============================================================
# YOLO-World v2 Small 在 DIOR 数据集上微调配置 (小白专用)
# ============================================================
# 数据集: DIOR (遥感卫星图像, 20 类, HBB 水平框)
# 模型: YOLO-World v2-S (Small, 显存需求小, 适合单卡训练)
# 训练: 80 epochs, AdamW, batch=8 (单卡 RTX 4090)
# ============================================================

_base_ = (
    '../../third_party/mmyolo/configs/yolov8/'
    'yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False)

# ============================================================
# 1. 超参数配置 (这里是最常需要改的地方)
# ============================================================
num_classes = 20                    # DIOR 20 类 (不要改)
num_training_classes = 20           # 训练时使用的类别数 (不要改)
max_epochs = 80                     # 训练轮数 (80 足够, 大数据集可减到 40)
close_mosaic_epochs = 10            # 最后 10 轮关闭 Mosaic 增强
save_epoch_intervals = 5            # 每 5 轮保存一次权重
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]

# 学习率和优化器 (显存不够就调小 batch, 学习率等比例缩小)
base_lr = 2e-4                      # AdamW 学习率 (SGD 用 1e-3)
weight_decay = 0.05
train_batch_size_per_gpu = 8        # 单卡 batch size (4090 用 8, 3090 用 4)

# 预训练模型 (需要提前下载放到 pretrained_models/ 目录)
load_from = 'pretrained_models/yolo_world_s_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_train-55b943ea.pth'
text_model_name = 'openai/clip-vit-base-patch32'

persistent_workers = False
mixup_prob = 0.15
copypaste_prob = 0.3

# ============================================================
# 2. 模型配置 (类别数自动跟随上面设置)
# ============================================================
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# ============================================================
# 3. 数据集配置 (DIOR 数据集路径, 根据你的实际路径修改)
# ============================================================
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
mosaic_affine_transform = [
    dict(
        type='MultiModalMosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(type='YOLOv5CopyPaste', prob=copypaste_prob),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=_base_.use_mask2refine)
]
train_pipeline = [
    *_base_.pre_transform,
    *mosaic_affine_transform,
    dict(
        type='YOLOv5MultiModalMixUp',
        prob=mixup_prob,
        pre_transform=[*_base_.pre_transform,
                       *mosaic_affine_transform]),
    *_base_.last_transform[:-1],
    *text_transform
]
train_pipeline_stage2 = [
    *_base_.train_pipeline_stage2[:-1],
    *text_transform
]

# 训练集配置
dior_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=[
            "airplane", "airport", "baseball-diamond", "basketball-court",
            "bridge", "chimney", "expressway-service-area", "expressway-toll-station",
            "dam", "golf-course", "ground-track-field", "harbor", "overpass",
            "ship", "stadium", "storage-tank", "tennis-court", "train-station",
            "vehicle", "windmill"
        ]),
        data_root='data/DIOR',                          # DIOR 数据根目录
        ann_file='annotations/train.json',              # 训练集 COCO 标注
        data_prefix=dict(img='images/train/'),           # 训练图片目录
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/dior_class_texts.json',  # 类别文本描述
    pipeline=train_pipeline)

train_dataloader = dict(
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=dior_train_dataset)

# 测试管道
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts'))
]

# 验证集配置
dior_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=[
            "airplane", "airport", "baseball-diamond", "basketball-court",
            "bridge", "chimney", "expressway-service-area", "expressway-toll-station",
            "dam", "golf-course", "ground-track-field", "harbor", "overpass",
            "ship", "stadium", "storage-tank", "tennis-court", "train-station",
            "vehicle", "windmill"
        ]),
        data_root='data/DIOR',
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/dior_class_texts.json',
    pipeline=test_pipeline)

val_dataloader = dict(dataset=dior_val_dataset)
test_dataloader = val_dataloader

# ============================================================
# 4. 训练设置
# ============================================================
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=3,             # 最多保留 3 个权重 (省硬盘)
        save_best='auto',             # 自动保存最佳模型
        interval=save_epoch_intervals))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])

# 优化器 (AdamW + 分层学习率, 文本编码器用更小学习率)
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={'backbone.text_model': dict(lr_mult=0.01),
                     'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')

# ============================================================
# 5. 评估设置
# ============================================================
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/DIOR/annotations/val.json',    # 验证集 COCO 标注
    metric='bbox')

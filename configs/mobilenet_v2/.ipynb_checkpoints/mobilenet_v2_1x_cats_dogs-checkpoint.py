# Inheritance
_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',
    '../_base_/schedules/imagenet_bs256_epochstep.py',
    '../_base_/default_runtime.py'
]

# ---- Model configs ----
# Here we use init_cfg to load pre-trained model.
# In this way, only the weights of backbone will be loaded.
# And modify the num_classes to match our dataset.

model = dict(
    backbone=dict(
        init_cfg = dict(
            type='Pretrained', 
            checkpoint='https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth', 
            prefix='backbone')
    ),
    head=dict(
        num_classes=2,
        topk = (1, )
    ))

# ---- Dataset configs ----
# We re-organized the dataset as ImageNet format.
dataset_type = 'ImageNet'
img_norm_cfg = dict(
     mean=[124.508, 116.050, 106.438],
     std=[58.577, 57.310, 57.437],
     to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    # Specify the batch size and number of workers in each GPU.
    # Please configure it according to your hardware.
    samples_per_gpu=32,
    workers_per_gpu=2,
    # Specify the training dataset type and path
    train=dict(
        type=dataset_type,
        data_prefix='data/cats_dogs_dataset/training_set/training_set',
        classes='data/cats_dogs_dataset/classes.txt',
        pipeline=train_pipeline),
    # Specify the validation dataset type and path
    val=dict(
        type=dataset_type,
        data_prefix='data/cats_dogs_dataset/val_set/val_set',
        ann_file='data/cats_dogs_dataset/val.txt',
        classes='data/cats_dogs_dataset/classes.txt',
        pipeline=test_pipeline),
    # Specify the test dataset type and path
    test=dict(
        type=dataset_type,
        data_prefix='data/cats_dogs_dataset/test_set/test_set',
        ann_file='data/cats_dogs_dataset/test.txt',
        classes='data/cats_dogs_dataset/classes.txt',
        pipeline=test_pipeline))

# Specify evaluation metric
evaluation = dict(metric='accuracy', metric_options={'topk': (1, )})

# ---- Schedule configs ----
# Usually in fine-tuning, we need a smaller learning rate and less training epochs.
# Specify the learning rate
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# Set the learning rate scheduler
lr_config = dict(policy='step', step=1, gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=2)

# ---- Runtime configs ----
# Output training log every 10 iterations.
log_config = dict(interval=10)

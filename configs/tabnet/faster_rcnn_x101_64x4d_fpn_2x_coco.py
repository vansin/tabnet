_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'

# Modify dataset related settings
dataset_type = 'CocoDataset'
# classes = ('table',)
data = dict(
    train=dict(
        img_prefix='/datasets/table0901/TableBank/Detection/images/',
        # classes=classes,
        ann_file='/datasets/table0901/TableBank/Detection/annotations/tablebank_word_train.json'),
    val=dict(
        img_prefix='/datasets/table0901/TableBank/Detection/images/',
        # classes=classes,
        ann_file='/datasets/table0901/TableBank/Detection/annotations/tablebank_word_val.json'),
    test=dict(
        img_prefix='/datasets/table0901/TableBank/Detection/images/',
        # classes=classes,
        ann_file='/datasets/table0901/TableBank/Detection/annotations/tablebank_word_test.json')
)

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))



# # We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=1),
#         mask_head=dict(num_classes=1)))


# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = None
# load_from = '/home/tml/vansin/lab-try/mmdetection/checkpoints/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
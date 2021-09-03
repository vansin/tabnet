# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))



# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('tabnet')
data = dict(
    train=dict(
        img_prefix='/datasets/table0901/TableBank/Detection/images/',
        classes=classes,
        ann_file='/datasets/table0901/TableBank/Detection/annotations/tablebank_word_train.json'),
    val=dict(
        img_prefix='/datasets/table0901/TableBank/Detection/images/',
        classes=classes,
        ann_file='/datasets/table0901/TableBank/Detection/images/tablebank_word_val.json'),
    test=dict(
        img_prefix='/datasets/table0901/TableBank/Detection/images/',
        classes=classes,
        ann_file='/datasets/table0901/TableBank/Detection/images/tablebank_word_test.json')
)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
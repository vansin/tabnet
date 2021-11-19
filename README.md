# tabnet

基于mmdetection的表格检测代码

## install

### Method 1

```shell
conda create -n tabnet python=3.7 -y
conda activate tabnet
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

pip install openmim
mim install mmdet

```
### Method 2

```shell
conda create -n tabnet_dev python=3.7 -y
conda activate tabnet_dev

# conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html



conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia -y
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html


pip install -r requirements/build.txt

pip install -v -e .  # or "python setup.py develop"

```


## 准备数据集

### 挂载群辉网络数据
```shell
sudo mount.cifs //v4.vansin.top/datasets /datasets -o user=vansin,pass=****,vers=2.0
sudo mount.cifs //192.168.4.21/datasets /datasets -o user=vansin,pass=****,vers=2.0

sudo mount -t cifs -o username=vansin,password=Tml768300.,uid=$(id -u),gid=$(id -g) //v4.vansin.top/vansin /home/tml/datasets
sudo mount -t cifs -o username=vansin,password=Tml768300.,uid=$(id -u),gid=$(id -g) //192.168.4.21/vansin /home/tml/datasets



mkdir /tmp/vansin
# sudo mount -t cifs -o username=vansin,password=Tml768300.,uid=$(id -u),gid=$(id -g) //192.168.4.21/vansin /tmp/vansin
sudo mount -t cifs -o username=vansin,password=Tml768300.,uid=$(id -u),gid=$(id -g) //v4.vansin.top/vansin /tmp/vansin
ln -s /tmp/vansin/datasets $(pwd)/data
ln -s /tmp/vansin/work_dirs $(pwd)/work_dirs
```
### 软链接
```shell
ln -s /datasets/table0901/TableBank/Detection/ /home/tml/vansin/tabnet/data/table
```
## Train
```shell
python tools/train.py configs/tabnet/table_v2.py
```
## Test
### Test 1 / 100 datasets
```shell
CUDA_VISIBLE_DEVICES=1 python test.py configs/tabnet/table_v4-mask_rcnn_swin-t-p4-w7_fpn_1x_coco_small.py \
/home/tml/vansin/paper/tabnet/work_dirs/table_v4-mask_rcnn_swin-t-p4-w7_fpn_1x_coco_small/epoch_12.pth \
--out results.pkl \
--eval bbox
```

### Test all the train

```shell
CUDA_VISIBLE_DEVICES=1 python test.py configs/tabnet/table_v4-mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py \
/home/tml/vansin/paper/tabnet/work_dirs/table_v4-mask_rcnn_swin-t-p4-w7_fpn_1x_coco/epoch_12.pth \
--out results.pkl \
--eval bbox
```



```shell
python demo/image_demo.py /datasets/table0901/TableBank/Detection/images/%5BMS-DOM2S%5D-180323_6.jpg \
    work_dirs/faster_rcnn_r50_fpn_2x_coco/table_v3.py \
    work_dirs/faster_rcnn_r50_fpn_2x_coco/epoch_2.pth \
    --device cpu
```


```shell
ln -s /tmp/ramdisk/Detection/ /home/tml/vansin/paper/tabnet/data/table

ln -s /run/user/1000/gvfs/smb-share:server=192.168.4.21,share=datasets/table0901/TableBank/Detection /home/tml/vansin/paper/tabnet/data/table
```

## ONNX模型转换


```Shell
python tools/deployment/pytorch2onnx.py \
    configs/tabnet/table_v4-mask_rcnn_swin-t-p4-w7_fpn_1x_3000.py \
    checkpoints/tabnet/epoch_12.pth \
    --output-file checkpoints/tabnet/epoch_12.onnx \
    --input-img demo/demo.jpg \
    --test-img tests/data/color.jpg \
    --shape 608 608 \
    --show \
    --verify \
    --dynamic-export \
    --cfg-options \
      model.test_cfg.deploy_nms_pre=-1
```

## 加速技巧

### RAM虚拟硬盘

```shell

mkdir /tmp/vansin_ram
sudo mount -t tmpfs -o size=72G tmpfs /tmp/vansin_ram
ln -s /tmp/vansin_ram/datasets $(pwd)/data

```

## 结果分析技巧

### 结果分析

```shell
python tools/analysis_tools/analyze_results.py \
configs/tabnet/table_v7-mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_icdar2013.py \
work_dirs/table_v7-mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_icdar2013/epoch_34.pkl \
results
```

### 网络图可视化

```shell
python tools/deployment/pytorch2onnx.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --output-file ${OUTPUT_FILE} \
    --input-img ${INPUT_IMAGE_PATH} \
    --shape ${IMAGE_SHAPE} \
    --test-img ${TEST_IMAGE_PATH} \
    --opset-version ${OPSET_VERSION} \
    --cfg-options ${CFG_OPTIONS}
    --dynamic-export \
    --show \
    --verify \
    --simplify
```

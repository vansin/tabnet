# tabnet

基于mmdetection的表格检测代码

## install

```shell
conda create -n tabnet python=3.7 -y
conda activate tabnet
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

pip install openmim
mim install mmdet

```


## 准备数据集

### 挂载群辉网络数据
```shell
sudo mount.cifs //v4.vansin.top/datasets /datasets -o user=vansin,pass=****,vers=2.0 
sudo mount.cifs //192.168.4.21/datasets /datasets -o user=vansin,pass=****,vers=2.0 
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

```shell
ln -s 
```

```shell
python demo/image_demo.py /datasets/table0901/TableBank/Detection/images/%5BMS-DOM2S%5D-180323_6.jpg \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --device cpu
```

```shell
python demo/image_demo.py /datasets/table0901/TableBank/Detection/images/%5BMS-DOM2S%5D-180323_6.jpg \
    work_dirs/faster_rcnn_r50_fpn_2x_coco/table_v3.py \
    work_dirs/faster_rcnn_r50_fpn_2x_coco/epoch_2.pth \
    --device cpu
```
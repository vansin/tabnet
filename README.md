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

```shell
ln -s 
```
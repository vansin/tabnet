#! /bin/bash

mkdir /tmp/vansin_ram

rm -rf data
rm -rf work_dirs

sudo mount -t tmpfs -o size=72G tmpfs /tmp/vansin_ram

cp /media/tml/data_vansin/datasets.zip /tmp/vansin_ram/datasets.zip
unzip /tmp/vansin_ram/datasets.zip -d /tmp/vansin_ram
ln -s /tmp/vansin_ram/datasets $(pwd)/data
rm -rf /tmp/vansin_ram/datasets.zip

ln -s /media/tml/data_tml/weight_files/tabnet/work_dirs $(pwd)/work_dirs
ln -s /media/tml/data_tml/inference_results/tabnet/results $(pwd)/results

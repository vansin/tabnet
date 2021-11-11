
mkdir /tmp/vansin_ram
sudo mount -t tmpfs -o size=72G tmpfs /tmp/vansin_ram


cp /home/tml/datasets/datasets.zip /tmp/vansin_ram/datasets.zip


tar -xvf /tmp/vansin_ram/datasets.zip


ln -s /tmp/vansin_ram/datasets $(pwd)/data

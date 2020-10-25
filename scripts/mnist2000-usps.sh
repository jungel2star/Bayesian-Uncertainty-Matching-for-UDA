#!/bin/bash

# abort entire script on error
set -e
export PYTHONPATH="$PWD:$PYTHONPATH"
#########mnist#####
python3  tools/uncertainDA_train_open.py mnist2000:train usps1800:train lenet2 uncertainDA_mnist2usps \
       --iterations 10000 \
       --batch_size 128 \
       --display 50 \
       --lr 0.001 \
       --snapshot 10000 \
       --weights snapshot/uncertainDA_mnist2usps\
       --adversary_relu \
       --solver Adam  \
       --netvladflag 1

#########usps#####
python3 tools/uncertainDA_train_open.py usps1800:train mnist2000:train lenet uncertainDA_usps2mnist \
       --iterations 10000 \
       --batch_size 256 \
       --display 50 \
       --lr 0.001 \
       --snapshot 10000 \
       --weights snapshot/uncertainDA_usps2mnist\
       --adversary_relu \
       --solver adam  \
       --netvladflag 1

python3 tools/eval_uncertainDA.py usps1800 test lenet2 snapshot/uncertainDA_mnist2usps --netvladflag 1
python3 tools/eval_uncertainDA.py mnist2000 train lenet2 snapshot/uncertainDA_usps2mnist --netvladflag 1

#!/bin/bash

# abort entire script on error
set -e

export PYTHONPATH="$PWD:$PYTHONPATH"

#########usps#####
python3.5 tools/uncertainDA_train.py mnist:train usps:train lenet2 uncertainDA_mnist2usps2 \
       --iterations 7000 \
       --batch_size 64 \
       --display 200 \
       --lr 0.001 \
       --snapshot 3500 \
       --weights snapshot/uncertainDA_mnist2usps\
       --adversary_relu \
       --solver Adam  \
       --netvladflag 1

python3.5 tools/uncertainDA_train.py mnist2000:train usps:train lenet uncertainDA_mnist2usps22 \
       --iterations 4500 \
       --batch_size 128 \
       --display 50 \
       --lr 0.001 \
       --snapshot 4500 \
       --weights snapshot/uncertainDA_mnist2usps\
       --adversary_relu \
       --solver adam  \
       --netvladflag 1


python3.5 tools/uncertainDA_train.py mnist2000:train usps:train lenet uncertainDA_mnist2usps222 \
       --iterations 4500 \
       --batch_size 128 \
       --display 50 \
       --lr 0.001 \
       --snapshot 4500 \
       --weights snapshot/uncertainDA_mnist2usps\
       --adversary_relu \
       --solver adam  \
       --netvladflag 1


python3.5 tools/uncertainDA_train.py mnist2000:train usps:train lenet uncertainDA_mnist2usps2222 \
       --iterations 4500 \
       --batch_size 128 \
       --display 50 \
       --lr 0.001 \
       --snapshot 4500 \
       --weights snapshot/uncertainDA_mnist2usps\
       --adversary_relu \
       --solver adam  \
       --netvladflag 1

python3.5 tools/eval_uncertainDA.py usps test lenet2 snapshot/uncertainDA_mnist2usps2 --netvladflag 1
python3.5 tools/eval_uncertainDA.py usps train lenet2 snapshot/uncertainDA_mnist2usps22 --netvladflag 1
python3.5 tools/eval_uncertainDA.py usps train lenet2 snapshot/uncertainDA_mnist2usps222 --netvladflag 1
python3.5 tools/eval_uncertainDA.py usps train lenet2 snapshot/uncertainDA_mnist2usps2222 --netvladflag 1

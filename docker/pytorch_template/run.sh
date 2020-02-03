DATADIR=/home/ubuntu/DataLake/Data/CIFAR10DEBUG
p=5e34e9021c9d44000054264e
e=5e3668771c9d44000088c11d
EXPTDIR=/home/ubuntu/DataLake/UserProjects/$p/$e

docker run -it  --mount type=bind,source=$DATADIR,dst=/data \
    --mount type=bind,source=$EXPTDIR,dst=/log --gpus device=0 \
    -e TASK='train' -e CUDA_DEVICE='cuda:0' -e DATA_DIR='/data' \
    -e EXPT_DIR=/log -e LOOP=0 hongshan1989/pytorch_resnet:example bash

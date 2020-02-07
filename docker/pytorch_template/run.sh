DATADIR=/home/ubuntu/DataLake/Data/CIFAR10DEBUG
EXPTDIR=$EPXT_DIR
echo $EXPT_DIR

docker run -it  --mount type=bind,source=$DATADIR,dst=/data \
    --mount type=bind,source=$EXPT_DIR,dst=/log --gpus device=0 \
    -e TASK='train' -e CUDA_DEVICE='cuda:0' -e DATA_DIR='/data' \
    -e EXPT_DIR=/log -e CKPT_FILE=ckpt_0.pth alectio/pytorch_resnet:example bash

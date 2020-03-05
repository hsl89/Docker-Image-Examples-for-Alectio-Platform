DATADIR=/home/ubuntu/DataLake/Data/DailyDialog
EXPT_DIR=/home/ubuntu/Common/tmp
VECTOR_DIR=/home/ubuntu/DataLake/Vector

docker run -it  --mount type=bind,source=$DATADIR,dst=/data \
    --mount type=bind,source=$EXPT_DIR,dst=/log --gpus device=0 \
    --mount type=bind,source=$VECTOR_DIR,dst=/vector\
    -e TASK='train' -e CUDA_DEVICE='cuda:0' -e DATA_DIR='/data' \
    -e VECTOR_DIR='/vector' \
    -e EXPT_DIR=/log -e CKPT_FILE=ckpt alectio/dailydialog_lstm:latest bash

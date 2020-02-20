import os
import docker

data_dir = '/home/ubuntu/DataLake/Data/COCO2014DEBUG'
expt_dir = '/home/ubuntu/Common/tmp'

client = docker.from_env()

client.containers.run('alectio/coco_yolov3:latest', 'python main.py', 
        volumes={
            expt_dir: {'bind': '/log', 'mode':'rw'},
            data_dir: {'bind': '/data', 'mode': 'ro'},
            }, 
        environment={
            'TASK': 'infer',
            'CUDA_DEVICE': 'cuda:0',
            'DATA_DIR': '/data',
            'EXPT_DIR': '/log',
            'CKPT_FILE': 'ckpt_0',
            'RESUME_FROM': 'ckpt_49.pt'
            },
        user=1000,
        runtime='nvidia')

        

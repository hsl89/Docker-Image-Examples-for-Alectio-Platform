import os
import docker

data_dir = '/home/ubuntu/DataLake/Data/CIFAR10DEBUG'
expt_dir = '/home/ubuntu/DataLake/UserProjects/5e34e9021c9d44000054264e/baseline'

client = docker.from_env()

client.containers.run('alectio/pytorch_resnet:example', 'python main.py', 
        volumes={
            expt_dir: {'bind': '/log', 'mode':'rw'},
            data_dir: {'bind': '/data', 'mode': 'ro'},
            }, 
        environment={
            'TASK': 'infer',
            'CUDA_DEVICE': 'cuda:0',
            'DATA_DIR': '/data',
            'EXPT_DIR': '/log',
            'CKPT_FILE': 'ckpt.pth',
            },
        user=1000,
        runtime='nvidia')

        

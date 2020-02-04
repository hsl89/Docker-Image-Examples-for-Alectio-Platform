import os
import docker

data_dir = '/home/ubuntu/DataLake/Data/CIFAR10DEBUG'
expt_dir = '/home/ubuntu/DataLake/UserProjects/5e34e9021c9d44000054264e/5e3668771c9d44000088c11d'

client = docker.from_env()

client.containers.run('alectio/pytorch_resnet:example', 'python main.py', 
        volumes={
            expt_dir: {'bind': '/log', 'mode':'rw'},
            data_dir: {'bind': '/data', 'mode': 'ro'},
            }, 
        environment={
            'TASK': 'train',
            'CUDA_DEVICE': 'cuda:0',
            'DATA_DIR': '/data',
            'EXPT_DIR': '/log',
            'LOOP': 0,
            },
        user=1000,
        runtime='nvidia')

        

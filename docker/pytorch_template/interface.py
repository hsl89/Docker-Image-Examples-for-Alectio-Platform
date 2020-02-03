import os
import docker

data_dir = '/home/ubuntu/Common/Data/CIFAR10DEBUG'
expt_dir = '/home/ubuntu/DataLake/UserProjects/5e34e9021c9d44000054264e/5e3668771c9d44000088c11d'

client = docker.from_env()

client.containers.run('hongshan1989/pytorch_resnet:example', 'python main.py', 
        volumes={
            expt_dir: {'bind': '/log', 'mode':'rw'},
            data_dir: {'bind': '/data', 'mode': 'rw'},
            }, 
        environment={
            'TASK': 'infer',
            'CUDA_DEVICE': 'cuda:0',
            'DATA_DIR': '/data',
            'EXPT_DIR': '/log',
            'LOOP': 0,
            }, 
        runtime='nvidia')

        

import os
import docker

data_dir = '/home/ubuntu/DataLake/Data/CIFAR10DEBUG'
expt_dir = '/home/ubuntu/Common/tmp'

client = docker.from_env()

def run(image):

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
                'CKPT_FILE': 'ckpt.pth',
                },
            user=1000,
            runtime='nvidia')
    return

nets = ['resnet', 'googlenet',
        'efficientnet', 'vgg']

for n in nets:
    image = f"alectio/cifar10_{n}:latest"
    run(image)

        

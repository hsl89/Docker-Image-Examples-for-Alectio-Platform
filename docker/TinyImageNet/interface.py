import os
import docker

data_dir = '/home/ubuntu/DataLake/Data/TinyImageNetDebug'
expt_dir = '/home/ubuntu/Common/tmp'

client = docker.from_env()

def run(image):
    print('Testing on {}'.format(image))
    client.containers.run(image, 'python main.py', 
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
    image = f"alectio/tinyimagenet_{n}:latest"
    run(image)

        

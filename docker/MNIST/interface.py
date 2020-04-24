import os
import docker

data_dir = '/home/ubuntu/DataLake/Data/MNIST'
expt_dir = '/home/ubuntu/Common/tmp'
client = docker.from_env()

def run(image):
    client.containers.run(image, 'python main.py', 
            volumes={
                expt_dir: {'bind': '/log', 'mode':'rw'},
                data_dir: {'bind': '/data', 'mode': 'ro'},
                }, 
            environment={
                'TASK': 'infer',
                'CUDA_DEVICE': 'cuda:0',
                'DATA_DIR': '/data',
                'EXPT_DIR': '/log',
                'CKPT_FILE': 'ckpt',
                },
            user=os.getuid(),
            runtime='nvidia')
    return


run('alectio/mnist_lenet:latest')
        

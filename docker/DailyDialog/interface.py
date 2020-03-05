import os
import docker

data_dir = '/home/ubuntu/DataLake/Data/DailyDialogDebug'
expt_dir = '/home/ubuntu/Common/tmp'
vector_dir = '/home/ubuntu/DataLake/Vector'

client = docker.from_env()

def run(image):

    client.containers.run(image, 'python main.py', 
            volumes={
                expt_dir: {'bind': '/log', 'mode':'rw'},
                vector_dir: {'bind': '/vector', 'mode': 'ro'},
                data_dir: {'bind': '/data', 'mode': 'ro'},
                }, 
            environment={
                'TASK': 'infer',
                'CUDA_DEVICE': 'cuda:0',
                'VECTOR_DIR': '/vector',
                'DATA_DIR': '/data',
                'EXPT_DIR': '/log',
                'CKPT_FILE': 'ckpt_0',
                },
            user=1000,
            runtime='nvidia')
    return


run('alectio/dailydialog_lstm:latest')


        

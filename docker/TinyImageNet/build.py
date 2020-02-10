import docker

nets = ['resnet', 'googlenet',
        'vgg', 'efficientnet']
client = docker.from_env()

# rebuild base
base_image = 'alectio/cifar10:latest'
client.images.build(path="./", dockerfile='./Dockerfile.cifar',
        tag=base_image)
client.images.push(base_image)


for n in nets:
    image = f"alectio/cifar10_{n}:latest"
    df=f"./Dockerfile.{n}"
    client.images.build(path="./", dockerfile=df, tag=image)
    client.images.push(image)



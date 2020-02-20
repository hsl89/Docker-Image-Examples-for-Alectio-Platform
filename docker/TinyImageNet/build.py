import docker

nets = ['resnet', 'googlenet',
        'vgg', 'efficientnet']
client = docker.from_env()

# rebuild base
base_image = 'alectio/tinyimagenet:latest'
client.images.build(path="./", dockerfile='./Dockerfile.ti',
        tag=base_image)
client.images.push(base_image)


for n in nets:
    image = f"alectio/tinyimagenet_{n}:latest"
    df=f"./Dockerfile.{n}"
    client.images.build(path="./", dockerfile=df, tag=image)
    client.images.push(image)



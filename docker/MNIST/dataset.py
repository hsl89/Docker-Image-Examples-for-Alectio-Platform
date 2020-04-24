import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import os


# setup dataset
data_dir = os.getenv('DATA_DIR') 



train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,))
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,))
    ])


mnist_train = MNIST(root=data_dir,  train=True,
        transform=train_transform)
# use for infer uncertainty, dont apply transform
mnist_infer = MNIST(root=data_dir,  train=True,
        transform=test_transform)

mnist_test = MNIST(root=data_dir,  train=False, 
        transform=test_transform)








if __name__ == "__main__":
    print(dir(mnist_train))
    print(len(mnist_train))
    for i in range(10):
        im, t = mnist_train[i]
        print(im.shape, t)

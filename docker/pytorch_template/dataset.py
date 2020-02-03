import torchvision.transforms as transforms
import os

from imagedata import ImageDataCLS

# setup dataset
data_dir = os.getenv('DATA_DIR') 


train_transform = transforms.Compose([
    # transforms.RandomRotation(degrees=[0, 360]),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914,0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010))
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914,0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010))
    ])


cifar_train = ImageDataCLS(root=data_dir,  train=True,
        transform=train_transform)
# use for infer uncertainty, dont apply transform
cifar_infer = ImageDataCLS(root=data_dir,  train=True,
        transform=test_transform)

cifar_test = ImageDataCLS(root=data_dir,  train=False, 
        transform=test_transform)








if __name__ == "__main__":
    print(len(cifar_train))
    for i in range(10):
        im, t = cifar_train[i]
        print(im.shape, t)

import torchvision.transforms as transforms
import os
import copy

from imagedata import ImageDataCLS

# setup dataset
data_dir = os.getenv('DATA_DIR') 


train_transform = transforms.Compose([
    # transforms.RandomRotation(degrees=[0, 360]),
    transforms.RandomCrop(64, padding=4),
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


ti_train = ImageDataCLS(root=data_dir,  train=True,
        transform=train_transform)
# use for infer uncertainty, dont apply transform
ti_infer = ImageDataCLS(root=data_dir,  train=True,
        transform=test_transform)

ti_test = ImageDataCLS(root=data_dir,  train=False, 
        transform=test_transform)



def write_debug_data_dir(dataset, src, target):
    dataset.transform = None
    for i in range(100):
        img, t, path = dataset[i]
        tpath = path.replace(src, target)
        tpath = tpath.split('/')
        name = tpath.pop()
        dirname = '/'.join(tpath)

        if not os.path.exists(dirname):
            os.makedirs(dirname)
        img.save(os.path.join(dirname, name))



        if dataset.train:
            label_file = os.path.join( 'train_labels.txt')
        else:
            label_file = os.path.join( 'test_labels.txt')
        
        meta_info = dataset.meta_info[i]

        ln = '{},{},{},{}\n'.format(meta_info['idx'], meta_info['image_name'],
                meta_info['wnid'], meta_info['label'])

        with open(label_file, 'a') as f:
            f.write(ln)


        

if __name__ == "__main__":
    write_debug_data_dir(ti_train, 'TinyImageNet', 'TinyImageNetDebug')
    write_debug_data_dir(ti_test, 'TinyImageNet', 'TinyImageNetDebug')


import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(imdb):
    #TODO: classes: list of classes
    #TODO: class_to_idx: dictionary with keys=classes and values=class index
    #If you did Task 0, you should know how to set these values from the imdb
    classes = imdb._classes
    class_to_idx = imdb._class_to_ind
    return classes, class_to_idx

def make_dataset(imdb, class_to_idx):
    #TODO: return list of (image path, list(+ve class indices)) tuples
    #You will be using this in IMDBDataset
    images = imdb.num_images
    path = [imdb.image_path_at(i) for i in range(images)]
    roidb = imdb.gt_roidb()
    class_indices = [list(set(roidb[i]['gt_classes']-1)) for i in range(images)]
    dataset_list = zip(path, class_indices)
    return dataset_list

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        #TODO: Define model
        # ceil_mode by default is false
        self.features_network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, dilation = 1),            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, dilation =1),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Classifier layer begins
        self.classifier_network = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 20, kernel_size=1, padding=1),
        )   
    
    def forward(self, x):
        #TODO: Define forward pass
        x = self.features_network(x)
        x = self.classifier_network(x)
        return x

class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        #TODO: Ignore for now until instructed
        self.features_network = nn.Sequential(
            nn.Conv2d(3, 64, (11, 11), (4, 4), (2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), (2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), (2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True))
        self.classifier_network = nn.Sequential(
            nn.Conv2d(256, 256, (3, 3), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 20, (1, 1), (1, 1)))


    def forward(self, x):
        #TODO: Ignore for now until instructed
        x = self.features_network(x)
        heatmap = self.classifier_network(x)

        avg1 = F.avg_pool2d(heatmap, (3,3))
        avg2 = F.avg_pool2d(avg1, (3,3), 2)
        m, n = heatmap.size()[2:]
        # print(m)
        max1 = F.max_pool2d(heatmap, (m,n))
        m, n = avg1.size()[2:]
        max2 = F.max_pool2d(avg1, (m,n))
        m, n = avg2.size()[2:]
        max3 = F.max_pool2d(avg2, (m,n))
        x = max1 + max2 + max3
        return x



def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    #TODO: Initialize weights correctly based on whethet it is pretrained or not

    if pretrained == True:
        print("load pretrained model")
        pretrained_state = model_zoo.load_url(model_urls['alexnet'].replace('https://', 'http://'))
        pretrained_state = {k: v for k, v in pretrained_state.items() if k.split('.')[0] == 'features'}
        model_state = model.state_dict()
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)

    for layer in model.classifier_network:
        if type(layer) == nn.Conv2d:
            nn.init.xavier_uniform(layer.weight)
            nn.init.xavier_uniform(layer.bias)



def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    #TODO: Ignore for now until instructed

    if pretrained == True:
        pretrained_state_ori = model_zoo.load_url(model_urls['alexnet'].replace('https://', 'http://'))
        pretrained_state = {k: v for k, v in pretrained_state.items() if k.split('.')[0] == 'features'}
        model_state = model.state_dict()
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)

        for layer in model.classifier_network:
            if type(layer) == nn.Conv2d:
                nn.init.xavier_uniform(layer.weight)


    return model




class IMDBDataset(data.Dataset):
    """A dataloader that reads imagesfrom imdbs
    Args:
        imdb (object): IMDB from fast-rcnn repository
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, list(+ve class indices)) tuples
    """

    def __init__(self, imdb, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(imdb)
        imgs = make_dataset(imdb, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images, what's going on?"))
        self.imdb = imdb
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a binary vector with 1s
                                   for +ve classes and 0s for -ve classes
                                   (it can be a numpy array)
        """
        # TODO: Write this function, look at the imagenet code for inspiration
        path, cls = self.imgs[index]
        img = self.loader(path)
        target = np.zeros((len(self.classes)))
        target[cls] = 1

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

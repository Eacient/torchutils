import random
import os.path
import PIL.Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as vision_tf
# need cls_label.npy need img_name_list
### need fix load function
"""main usage
- ClassificationDataset
    - require: img_id_list_file, transform
    - return: name, img, cls_label
- ClassificationDatasetWithSaliency
    crop_size=224, resize_size=(256, 512)
    - require: img_id_list_file, saliency_root
    - return: img_id, img, saliency[h,w], cls_label
"""

def load_img_id_list(img_id_file):
    return open(img_id_file).read().splitlines()


def load_img_label_list_from_npy(img_name_list, dataset):
    cls_labels_dict = np.load(f'metadata/{dataset}/cls_labels.npy', allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_saliency_path(img_name, saliency_root='SALImages'):
    return os.path.join(saliency_root, img_name + '.png')


class ImageDataset(Dataset):
    """
    Base image dataset. This returns 'img_id' and 'image'
    """
    def __init__(self, dataset, img_id_list_file, img_root, transform=None):
        self.dataset = dataset
        self.img_id_list = load_img_id_list(img_id_list_file)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.img_id_list)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = PIL.Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img_id, img


class ClassificationDataset(ImageDataset):
    """
    Classification Dataset (base)
    """
    def __init__(self, dataset, img_id_list_file, img_root, transform=None):
        super().__init__(dataset, img_id_list_file, img_root, transform)
        self.label_list = load_img_label_list_from_npy(self.img_id_list, dataset)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)
        label = torch.from_numpy(self.label_list[idx])
        return name, img, label


class ClassificationDatasetWithSaliency(ImageDataset):
    """
    Classification Dataset with saliency
    """
    def __init__(self, dataset, img_id_list_file, img_root, saliency_root=None, crop_size=224, resize_size=(256, 512)):
        super().__init__(dataset, img_id_list_file, img_root, transform=None)
        self.saliency_root = saliency_root
        self.crop_size = crop_size
        self.resize_size = resize_size

        self.resize = RandomResizeLong(resize_size[0], resize_size[1])
        self.color = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.normalize = Normalize()

        self.label_list = load_img_label_list_from_npy(self.img_id_list, dataset)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = PIL.Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")
        saliency = PIL.Image.open(get_saliency_path(img_id, self.saliency_root)).convert("RGB")
        img, saliency = self.transform_with_mask(img, saliency)

        label = torch.from_numpy(self.label_list[idx])
        return img_id, img, saliency, label

    def transform_with_mask(self, img, mask):
        # randomly resize
        target_size = random.randint(self.resize_size[0], self.resize_size[1])
        img = self.resize(img, target_size)
        mask = self.resize(mask, target_size)

        # randomly flip
        if random.random() > 0.5:
            img = vision_tf.hflip(img)
            mask = vision_tf.hflip(mask)

        # add color jitter
        img = self.color(img)

        img = np.asarray(img)
        mask = np.asarray(mask)

        # normalize
        img = self.normalize(img)
        mask = mask / 255.
        img, mask = random_crop_with_saliency(img, mask, self.crop_size)

        # permute the order of dimensions
        img = HWC_to_CHW(img)
        mask = HWC_to_CHW(mask)

        # make tensor
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        mask = torch.mean(mask, dim=0, keepdim=True)

        return img, mask
    


#----------------------------transforms


class RandomResizeLong:

    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, img, target_long=None, mode='image'):
        if target_long is None:
            target_long = random.randint(self.min_long, self.max_long)
        w, h = img.size

        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))

        if mode == 'image':
            img = img.resize(target_shape, resample=PIL.Image.CUBIC)
        elif mode == 'mask':
            img = img.resize(target_shape, resample=PIL.Image.NEAREST)

        return img


class RandomCrop:

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr):

        h, w, c = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        container = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            imgarr[img_top:img_top+ch, img_left:img_left+cw]

        return container


def random_crop_with_saliency(imgarr, mask, crop_size):

    h, w, c = imgarr.shape

    ch = min(crop_size, h)
    cw = min(crop_size, w)

    w_space = w - crop_size
    h_space = h - crop_size

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space+1)
    else:
        cont_left = random.randrange(-w_space+1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space+1)
    else:
        cont_top = random.randrange(-h_space+1)
        img_top = 0

    container = np.zeros((crop_size, crop_size, imgarr.shape[-1]), np.float32)
    container_mask = np.zeros((crop_size, crop_size, imgarr.shape[-1]), np.float32)
    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        imgarr[img_top:img_top+ch, img_left:img_left+cw]
    container_mask[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        mask[img_top:img_top+ch, img_left:img_left+cw]

    return container, container_mask


class RandomHorizontalFlip():
    def __init__(self):
        return

    def __call__(self, img):
        if bool(random.getrandbits(1)):
            img = np.fliplr(img).copy()
        return img


class CenterCrop():

    def __init__(self, cropsize, default_value=0):
        self.cropsize = cropsize
        self.default_value = default_value

    def __call__(self, npimg):

        h, w = npimg.shape[:2]

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        sh = h - self.cropsize
        sw = w - self.cropsize

        if sw > 0:
            cont_left = 0
            img_left = int(round(sw / 2))
        else:
            cont_left = int(round(-sw / 2))
            img_left = 0

        if sh > 0:
            cont_top = 0
            img_top = int(round(sh / 2))
        else:
            cont_top = int(round(-sh / 2))
            img_top = 0

        if len(npimg.shape) == 2:
            container = np.ones((self.cropsize, self.cropsize), npimg.dtype)*self.default_value
        else:
            container = np.ones((self.cropsize, self.cropsize, npimg.shape[2]), npimg.dtype)*self.default_value

        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            npimg[img_top:img_top+ch, img_left:img_left+cw]

        return container

def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_arr = np.asarray(img)
        normalized_img = np.empty_like(img_arr, np.float32)

        normalized_img[..., 0] = (img_arr[..., 0] / 255. - self.mean[0]) / self.std[0]
        normalized_img[..., 1] = (img_arr[..., 1] / 255. - self.mean[1]) / self.std[1]
        normalized_img[..., 2] = (img_arr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return normalized_img
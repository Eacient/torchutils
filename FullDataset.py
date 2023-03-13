from torch.utils.data import Dataset
import numpy as np
import imageio.v3 as imageio
from PIL import Image
import os
import torch
import random
"""main usage
    used for cam train and trainval
    train: random resize crop flip
    return: img_id, img, label, sal, seg
"""

IMG_FOLDER_NAME = "JPEGImages"
SAL_FOLDER_NAME = "SALImages"
SEG_FOLDER_NAME = "SegmentationClassAug"
IGNORE = 255


cls_labels_dict = np.load('metadata/voc12/cls_labels.npy', allow_pickle=True).item()

def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.int32)
    # img_name_list = np.loadtxt(dataset_path, dtype=str)
    # print(img_name_list[0])
    return img_name_list

def load_image_label_list_from_npy(img_name_list):
    return np.array([cls_labels_dict[decode_int_filename(img_name)] for img_name in img_name_list])

class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

def get_img_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def get_sal_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, SAL_FOLDER_NAME, img_name + '.png')

def get_seg_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, SEG_FOLDER_NAME, img_name + '.png')


class VOC12Dataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(),
                 crop_size=None, crop_method=None, to_torch=True):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root

        self.resize_long = resize_long
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))
        sal = np.asarray(imageio.imread(get_sal_path(name_str, self.voc12_root)))
        seg = np.asarray(imageio.imread(get_seg_path(name_str, self.voc12_root)))
        label = torch.from_numpy(self.label_list[idx])
    

        img, sal, seg = random_resize_long((img,sal,seg), self.resize_long[0], self.resize_long[1])
        img, sal, seg = random_lr_flip((img,sal,seg))
        img, sal, seg = random_crop((img,sal,seg), self.crop_size, [0,0,255])

        img = self.img_normal(img)

        img = HWC_to_CHW(img)
        sal = HWC_to_CHW(sal)
        sal = sal.mean(axis=0)
        return name_str, img, label, sal, seg


def HWC_to_CHW(img):
    return np.transpose(img, (2, 0, 1))


def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img

    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return np.asarray(Image.fromarray(img).resize(size[::-1], resample))

def pil_rescale(img, scale, order):
    height, width = img.shape[:2]
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)

def random_resize_long(img, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h, w = img[0].shape[:2]

    if w < h:
        scale = target_long / h
    else:
        scale = target_long / w

    return [pil_rescale(m, scale, 3) for m in img]
    

def random_lr_flip(img):
    return [np.fliplr(m) for m in img]

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def random_crop(images, cropsize, default_values):

    if isinstance(images, np.ndarray): images = (images,)
    if isinstance(default_values, int): default_values = (default_values,)

    imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, default_values):

        if len(img.shape) == 3:
            cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
        else:
            cont = np.ones((cropsize, cropsize), img.dtype)*f
        cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
        new_images.append(cont)

    return new_images

def bulid_voc_datasets(args):
    return VOC12Dataset(args.name_list, args.voc12_root, resize_long=args.resize_long, crop_size=args.crop_size)

if __name__ == "__main__":
    ds = VOC12Dataset('metadata/voc12/train_aug.txt', voc12_root='/home/dogglas/mil/datasets/VOC2012',
                      resize_long=(16, 24), crop_size=16)
    ds[0]
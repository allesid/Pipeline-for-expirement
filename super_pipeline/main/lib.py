
# from torch import tensor, broadcast_to
import sys
# sys.path.append(r"/home/alex/GitLabProjects/NN_pipeline/")
import nibabel as nib

from super_pipeline.dataload import augmentation as aug
import torch
import torchvision
import pydicom as dicom
import numpy as np
# from torchvision.transforms.functional import to_tensor
from torchvision.io import read_image
import yaml
import os
import random

def open_config_file(path2file):
    with open(path2file) as fh:
        ds = yaml.load(fh, Loader=yaml.FullLoader)
    return ds


def check_classes(ymdatafn):
    ymdataset = open_config_file('../configs/data/'+ymdatafn)

    data_path = ymdataset["data_path"]
    lp = []
    lpath = ["train_dir", "val_dir", "test_dir"]
    for p in lpath:
        lt = sorted(os.listdir(os.path.join(data_path, ymdataset[p])))
        lp.append(lt)
    if lp[0] == lp[1]:
        if lp[0] == lp[2]:
            return lp[0]
        else:
            print(
                f"Classes in {ymdataset[lpath[0]]} and {ymdataset[lpath[2]]} folders don't equal.")
            sys.exit(1)
    else:
        print(
            f"Classes in {ymdataset[lpath[0]]} and {ymdataset[lpath[1]]} folders don't equal.")
        sys.exit(1)


def conv_1img_totensor(img_path) -> torch.tensor:
    """
    Convert JPEG or DCM image file to pytorch tensor

    Args:
        img_path (str): path of the JPEG or PNG image.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    if img_path.endswith(".dcm"):
        ds = dicom.dcmread(img_path)
        img = ds.pixel_array
        # imgmax = 2500   # v1-v3 not v4
        # vfun = np.vectorize(lambda x: x if x <= imgmax else imgmax)   # v1-v3 not v4
        # img = vfun(img)   # v1-v3 not v4
        # vfun = np.vectorize(lambda x: x if x >= 0 else 0.0)
        # img = vfun(img) # .astype(float)
        # img = torch.tensor((img / img.max()) * 255, dtype=torch.uint8) # v1-v2
        img = np.clip(img, 0, 2550).astype(np.uint8)
        img = torch.tensor(img, dtype=torch.uint8)  # v3-v4
    elif img_path.endswith((".jpg", ".png", ".jpeg")):
        img = read_image(img_path)
        img = torch.squeeze(img)
    # print(image.shape)
    elif img_path.endswith((".nii", ".nii.gz")):
        img3d = nib.load(img_path).get_fdata()
        img = img3d.permute(2, 1, 0)
    else:
        return None
    if len(img.shape) == 2:
        img = torch.broadcast_to(img, (3,)+tuple(img.shape))

    return img


def conv_1img_3D_totensor(img_path) -> torch.tensor:
    """
    Convert NIFTI or DICOM 3D image file to pytorch tensor

    Args:
        img_path (str): path of the nii or dcm image.

    Returns:
        output (Tensor[image_channels, image_height, image_width])
    """
    if img_path.endswith(".dcm"):
        ds = dicom.dcmread(img_path)
        img = ds.pixel_array
        # imgmax = 2500   # v1-v3 not v4
        # vfun = np.vectorize(lambda x: x if x <= imgmax else imgmax)   # v1-v3 not v4
        # img = vfun(img)   # v1-v3 not v4
        # vfun = np.vectorize(lambda x: x if x >= 0 else 0.0)
        # img = vfun(img) # .astype(float)
        # img = torch.tensor((img / img.max()) * 255, dtype=torch.uint8) # v1-v2
        img = np.clip(img, 0, 2550).astype(np.uint8)
        img = torch.tensor(img, dtype=torch.uint8)  # v3-v4
    # print(image.shape)
    elif img_path.endswith((".nii", ".nii.gz")):
        img3d = nib.load(img_path).get_fdata()
        img = np.clip(img3d, -350, 350).astype(np.uint8)
        img = torch.tensor(img, dtype=torch.uint8)  # v3-v4
        img = img.permute(2, 1, 0)
    else:
        return None

    return img

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp",
                  ".pgm", ".tif", ".tiff", ".webp", ".dcm", ".nii", ".nii.gz")


class ImageFolderDCM(torchvision.datasets.DatasetFolder):
    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
        loader=None,
        is_valid_file=None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples


def img_normalize(img) -> torch.Tensor:
    """
    Normalize image

    Args:
    img: (Tensor) - raw image

    Return:
    Normalized tensor with 0-1 values
    """
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))    
    # (img - torch.mean(img)) / torch.std(img, False)

def rseed(seed):
    """
    Specify seed

    Args:
    img: seed - int

    Return:
    None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def data_loader(dataset_yaml, t_dir, t_transforms, shuffle, batch_size):
    """
    Determine DataLoader for specific folder

    Args:
    dataset_yaml - str, yaml file name in ../config/data/, specified dataset paths
    t_dir - str,  folder name with data
    t_transforms - torchvision.transforms, data transforms
    shuffle - bool 
    batch_size - int, batch size

    Return:
    test_dataloader - torch.utils.data.DataLoader
    """
    datafolder = open_config_file('../configs/data/'+dataset_yaml)
    data_path = datafolder["data_path"]
    test_dir = datafolder[t_dir]

    test_dir = os.path.join(data_path, test_dir)

    test_dataset = ImageFolderDCM(
        test_dir, transform=t_transforms, loader=conv_1img_totensor)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=shuffle, num_workers=batch_size//2)
    return test_dataloader


def testloadconfig(configfilename):
    """
    Determine DataLoader for specific folder

    Args:
    configfilename - str, yaml file name in ../config/, determine train configuration

    Return:
    test_dir - str, full name for test data folder 
    ymargs - dict, args in arg.yaml file
    ymconfig - dict, args in config.yaml file
    test_dataloader - torch.utils.data.DataLoader
    """
    ymconfig = open_config_file(
        os.path.join('../configs/', configfilename))
    ymdatafn = ymconfig["dataset_conf"]
    ymargs = open_config_file(ymconfig["args"])

    seed = ymconfig["seed"]
    rseed(seed)

    trans = getattr(aug, ymconfig["augmentation"])
    test_transforms = trans("test")

    datafolder = open_config_file('../configs/data/'+ymdatafn)

    test_dir = os.path.join(datafolder["data_path"], datafolder["test_dir"])

    test_dataset = ImageFolderWithPaths(
        test_dir, transform=test_transforms, loader=conv_1img_totensor)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=ymargs["batch_size"], shuffle=False, num_workers=0)

    return test_dir, ymargs, ymconfig, test_dataloader


def sortload(configfilename):
    ymconfig = open_config_file(
        os.path.join('../configs/', configfilename))
    ymdatafn = ymconfig["dataset_conf"]
    ymargs = open_config_file(ymconfig["args"])

    seed = ymconfig["seed"]
    rseed(seed)

    trans = getattr(aug, ymconfig["augmentation"])
    test_transforms = trans("test")

    datafolder = open_config_file('../configs/data/'+ymdatafn)

    sort_dir = os.path.join(datafolder["data_path"], datafolder["sort_dir"])

    sort_dataset = ImageFolderWithPaths(
        sort_dir, transform=test_transforms, loader=conv_1img_totensor)

    test_dataloader = torch.utils.data.DataLoader(
        sort_dataset, batch_size=ymargs["batch_size"], shuffle=False, num_workers=0)

    return sort_dir, ymargs, ymconfig, test_dataloader


class ImageFolderWithPaths(ImageFolderDCM):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



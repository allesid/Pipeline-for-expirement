# Convert 2D dcm images to jpg images

import pydicom as dicom
import os
import cv2
import pandas as pd
import numpy as np
import sys
from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from fastai.medical.imaging import *

def rectangle(img, t_start, t_end, colour):
    txstart = int(min(t_start[0], t_end[0]))
    txend = int(max(t_start[0], t_end[0]))
    tystart = int(min(t_start[1], t_end[1]))
    tyend = int(max(t_start[1], t_end[1]))
    for x in [txstart, txend]:
        for i in range(tystart, tyend+1):
            img[i, x] = colour
    for y in [tystart, tyend]:
        for i in range(txstart, txend+1):
            img[y, i] = colour
    return img


def convert2D_dcm2jpg(dcm_folder_path="DCM_imgs_2D", jpg_folder_path="JPG_imgs_2D",
 img_info_fn="Images2D_information.csv"):

    """
    Read images in 2D .dcm format and convert them into .jpg images.
    Also extract image information from .dcm image into .csv file.

    Inputs:
    dcm_folder_path:  Specify the folder path for .dcm 2D images. Default: "DCM_imgs_2D".
    jpg_folder_path:  Specify the folder path for .jpg 2D images. Default: "JPG_imgs_2D".
                        If jpg_folder_path is missing, create default folder.
    img_info_fn:         Specify name for image information file. Default: "Images2D_information.csv"

    Returns:
    None
    """


    if not os.path.exists(jpg_folder_path):
        print(
            f"Folder path for .jpg 2D images is missing. Creating one: {jpg_folder_path}")
        os.mkdir(jpg_folder_path)

    annotation_file = "CrowdsCureCancer2017Annotations.csv"
    if os.path.exists(annotation_file):
        annots = pd.read_csv(annotation_file)
    else:
        print(f"File {annotation_file} is absent.")
        annots = None

    # Conversion
    items = get_dicom_files(dcm_folder_path)
    params = pd.DataFrame.from_dicoms(items)
    params.to_csv(os.path.join(jpg_folder_path, img_info_fn))
    # print(params['SOPInstanceUID'])
    for i in range(len(params)):
        image = params.loc[i,"fname"]
        xray_sample = items[i].dcmread()
        # print(xray_sample.pixel_array, xray_sample.pixel_array.shape)
        xray_sample.show()
        if image.endswith(".dcm"):
            sopii = None
            ds = dicom.dcmread(image)
            img = ds.pixel_array
            if len(img.shape) == 2:
                if annots is not None:
                    if params.loc[i, "SOPInstanceUID"] in list(annots["instanceUID"]):
                        sopii = params.loc[i, "SOPInstanceUID"]
                        # print('sopii=', sopii, ' type sopii=', type(sopii))
                        ann = annots[annots["instanceUID"]
                                    == sopii].reset_index()
                        # print(ann)
                        for ik in range(len(ann)):
                            xstart = ann.loc[ik, "start_x"]
                            ystart = ann.loc[ik, "start_y"]
                            xend = ann.loc[ik, "end_x"]
                            yend = ann.loc[ik, "end_y"]
                            # print("xy=", xstart, ystart, xend, yend)
                            img = rectangle(img, (xstart, ystart),
                                            (xend, yend), img.max())
                imgmax = 2550   # v1-v3 not v4
                img = img.clip(0, imgmax)  # v1
                img = (img / img.max()) * 255
                # img = (img - img.min()) / (img.max() - img.min()) * 255
                image = image.split('/')[-1].replace('.dcm', '')
                cv2.imwrite(os.path.join(jpg_folder_path,
                            image+'.jpg'), img)
            elif len(img.shape) == 3:
                print("3D DICOM files not provided now")
                pass
            else:
                print(f"Input error: {image} file is {len(img.shape)} dimentional, \
                expected 2 or 3 dimentions,\nfile is missing")
                continue
            print(
                f"{ds.filename} converted to jpg file, img.min= {img.min()}, img.max= {img.max()}")
            params = params.dropna(axis=1, how='all')
            # information about images saved in file Images2D_information.csv


def take_dcm_files(dcm_folder_path="DCM_imgs", jpg_folder_path="JPG_imgs", conv_fun=convert2D_dcm2jpg):
    """
    Takes all DICOM files recurcsively and convert to JPG files.

    Inputs:
    dcm_folder_path:  Specify the folder path for .dcm 2D images. Default: "DCM_imgs".
    jpg_folder_path:  Specify the folder path for .jpg 2D images. Default: "JPG_imgs".
                        If jpg_folder_path is missing, create default folder.
    conv_fun:         Specify conversion function. Default: "convert2D_dcm2jpg"

    Output:
    Files in jpg format

    Returns:
    None
   
    Directories structure in jpg_folder_path is the same as in dcm_folder_path
    """
    if not os.path.exists(dcm_folder_path):
        print(
            f"Folder path for .dcm 2D images is missing. Create {dcm_folder_path} folder and copy .dcm 2D images into it")
        sys.exit(0)
# os.makedirs(name, mode=0o777, exist_ok=True)

    with os.scandir(dcm_folder_path) as it:
        for entry in it:
            if entry.is_dir():
                dcm_fpath = os.path.join(dcm_folder_path, entry.name)
                jpg_fpath = os.path.join(jpg_folder_path, entry.name)
                os.makedirs(jpg_fpath, exist_ok=True)
                take_dcm_files(dcm_fpath, jpg_fpath, conv_fun)
            elif not entry.name.startswith('.') and entry.is_file():
                if entry.name.endswith('.dcm'):
                    conv_fun(dcm_folder_path, jpg_folder_path)
                    return
                else:
                    continue

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dcmpath', default="DCM_imgs_2D",
                        help='Specify the folder path for .dcm 2D images. Default: "DCM_imgs_2D".')
    parser.add_argument('-j', '--jpgpath', default="JPG_imgs_2D",
                        help='Specify the folder path for .jpg 2D images. Default: "JPG_imgs_2D".')
    parser.add_argument('-i', '--info', default="Images2D_information.csv",
                        help='Specify name for image information file. Default: "Images2D_information.csv"')
    args = parser.parse_args()
    dcm_folder_path = args.dcmpath
    jpg_folder_path = args.jpgpath
    img_info_fn = args.info
    take_dcm_files(dcm_folder_path, jpg_folder_path, conv_fun=convert2D_dcm2jpg)
    # convert2D_dcm2jpg(dcm_folder_path, jpg_folder_path, img_info_fn)

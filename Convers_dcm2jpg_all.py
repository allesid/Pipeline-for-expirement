# Convert 2D dcm images to jpg images

import pydicom as dicom
import os
import cv2
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm


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
    File "dicom_image_description.csv" consist of all available dicom image attributes and must be present in scripts directory.

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

    # file "dicom_image_description.csv" consist of all available dicom image attributes
    dicom_image_description_file = "dicom_image_description_short.csv"

    annotation_file = "CrowdsCureCancer2017Annotations.csv"
    if os.path.exists(annotation_file):
        annots = pd.read_csv(annotation_file)
    else:
        print(f"File {annotation_file} is absent.")
        annots = None

    if os.path.exists(dicom_image_description_file):
        params = pd.read_csv(dicom_image_description_file)
        lp = len(params.columns)
        pc = params.columns
    else:
        print(f"File {dicom_image_description_file} is absent.")
        sys.exit(0)

    # Conversion
    images_path = os.listdir(dcm_folder_path)
    for image in images_path:
        if image.endswith(".dcm"):
            sopii = None
            ds = dicom.dcmread(os.path.join(dcm_folder_path, image))
            # print(ds.top())
            # print(ds.values())
            img = ds.pixel_array
            # print('pixel_array_numpy.shape=', pixel_array_numpy.shape)
                # print('img=\n', img)
            if len(img.shape) == 2:
            # print("name: ", image)
            # img -= img[0, 0]
                lp = len(params)
                # print('lp=', lp)
                # extract image information
                for field in pc:
                    # print('field0=', field)
                    try:
                        if ds.data_element(field) is None:
                            params.loc[lp, field] = None
                        else:
                            s1 = str(ds.data_element(field)).replace("'", "")
                            s2 = s1.find(":")
                            s1 = s1[s2+2:]
                            params.loc[lp, field] = s1
                    except:
                        params.loc[lp, field] = None
                if annots:
                    if params.loc[lp, "SOPInstanceUID"] in list(annots["instanceUID"]):
                        sopii = params.loc[lp, "SOPInstanceUID"]
                        print('sopii=', sopii, ' type sopii=', type(sopii))
                        ann = annots[annots["instanceUID"]
                                    == sopii].reset_index()
                        print(ann)
                        for i in range(len(ann)):
                            xstart = ann.loc[i, "start_x"]
                            ystart = ann.loc[i, "start_y"]
                            xend = ann.loc[i, "end_x"]
                            yend = ann.loc[i, "end_y"]
                            print("xy=", xstart, ystart, xend, yend)
                            img = rectangle(img, (xstart, ystart),
                                            (xend, yend), img.max())
                imgmax = 2000
                vfun = np.vectorize(lambda x: x if x <= imgmax else imgmax)
                img = vfun(img)
                img = (img / img.max()) * 255
                # img = (img - img.min()) / (img.max() - img.min()) * 255
                image = image.replace('.dcm', '')
                cv2.imwrite(os.path.join(jpg_folder_path,
                            image+'.jpg'), img)
            elif len(img.shape) == 3:
                pass
            else:
                print(f"Input error: {image} file is {len(img.shape)} dimentional, \
                expected 2 or 3 dimentions,\nfile is missing")
                continue
            print(
                f"{ds.filename} converted to jpg file, img.min= {img.min()}, img.max= {img.max()}")
            params = params.dropna(axis=1, how='all')
            # information about images saved in file Images2D_information.csv
            params.to_csv(os.path.join(jpg_folder_path,
                          ds.PatientID + "-" + img_info_fn))


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

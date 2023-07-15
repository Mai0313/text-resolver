import datetime
import glob
import os
import shutil

import autoroot
import numpy as np
import pandas as pd
from PIL import Image


def get_seperated_data(data_path, mask_path, target_path):
    """
    Moving all files into case_* folders
    each folder will have nails.png and nails_mask.png
    """
    filenames = list(os.listdir(data_path))
    for filename in filenames:
        case_num = filename.split(".")[0]
        os.makedirs(f"{target_path}/info/case_{case_num}")
        shutil.copy(f"{data_path}/{filename}",
                    f"{target_path}/info/case_{case_num}/nails.png")
        shutil.copy(f"{mask_path}/{filename}",
                    f"{target_path}/info/case_{case_num}/nails_mask.png")
    print("Done")


def get_train_data(path):
    """
    Get train data case id
    """
    case = list(range(1, 54))
    data = pd.DataFrame([case])
    data.to_csv(f"{path}/train_cases.csv", index=None, header=None)


def get_val_data(path):
    """
    Get validation data case id
    """
    case = list(range(54, 64))
    data = pd.DataFrame([case])
    data.to_csv(f"{path}/val_cases.csv", index=None, header=None)


def main(tags):
    sep_date = datetime.datetime.now().strftime("%Y%m%d")
    target_path = f"./data/{sep_date}_nails"
    data_path = f"./data/{tags}/images"
    mask_path = f"./data/{tags}/masks"
    os.makedirs(target_path, exist_ok=True)
    get_train_data(target_path)
    get_val_data(target_path)
    get_seperated_data(data_path, mask_path, target_path)


def gen_data(data_dir):
    """
    This function includes three things: Parsing data to separate data, parse data.
    """
    case_folders = glob.glob(f"{data_dir}/case_*")

    for case_folder in case_folders:

        case_id = int(case_folder.split("_")[-1])

        nails_img = Image.open(f"{case_folder}/nails.png")
        nails_array = np.array(nails_img)

        nails_mask_img = Image.open(f"{case_folder}/nails_mask.png")
        nails_mask_array = np.array(nails_mask_img)

        np.savez(f"{case_folder}/nails.npz",
                 nails=nails_array,
                 nails_mask=nails_mask_array,
                 case_id=case_id)


if __name__ == "__main__":
    main("raw_data")
    gen_data("./data/20230716_nails/info")

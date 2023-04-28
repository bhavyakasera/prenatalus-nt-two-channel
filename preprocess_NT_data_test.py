import glob
import os
import numpy as np
import os
import pathlib

import pandas as pd
import numpy as np
import skimage.transform as st
from pathlib import Path
import cv2
from PIL import Image

DATA_FOLDER = '/Users/bhavyakasera/Documents/PrenatalUS/cleaned_and_cropped_test'
IMAGE_FOLDER = '/Users/bhavyakasera/Documents/PrenatalUS/data/prenatalUS_NT_full'
DATASHEET_PATH = '/Users/bhavyakasera/Documents/PrenatalUS/prenatal_data_info.csv'

df = pd.read_csv(DATASHEET_PATH)


def spacer():
    print('-' * 50)


def get_mask(file_name, NT=False):
    # PIL supported image types
    img_types = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]

    # print("Filename: ", file_name)
    if file_name[-3:].lower() in img_types or file_name[-4:].lower() in img_types:

        img = cv2.imread(file_name)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if not NT:
            lower = np.array([175, 50, 20])  # example value
            upper = np.array([180, 255, 255])  # example value
        else:
            lower = np.array([50, 100, 100])
            upper = np.array([70, 255, 255])

        mask = cv2.inRange(img_hsv, lower, upper)
        seg_map = cv2.bitwise_and(img, img, mask=mask)

        # seg_map = st.resize(seg_map, (512, 512))
        seg_map = seg_map[:, :, 2]

        return seg_map


patients = os.listdir(DATA_FOLDER)
data = []

for patient in patients:
    print('patient = ', patient)
    patient_num = patient.split(' ')[1]
    patient_path = os.path.join(DATA_FOLDER, patient)
    try:
        approved_ext = False
        extensions = glob.glob(f'{patient_path}/{patient_num}.3_cropped.*')
        for ext in extensions:
            image_extension = os.path.splitext(ext)[1]
            if image_extension.lower() in ['.jpg', '.tif', '.jpeg', '.tiff', '.png']:
                approved_ext = True
        print(f'found approved ext for {patient}? ', str(approved_ext))

        approved_ext_ann = False
        annotated_extensions = glob.glob(f'{patient_path}/{patient_num}.3_cropped_headseg.*')
        for ext in annotated_extensions:
            image_extension = os.path.splitext(ext)[1]
            if image_extension.lower() in ['.jpg', '.tif', '.jpeg', '.tiff', '.png']:
                approved_ext_ann = True
        print(f'found approved ext annotated for {patient}? ', str(approved_ext_ann))

        approved_ext_nt = False
        annotated_extensions = glob.glob(f'{patient_path}/{patient_num}.3_cropped_NT.*')
        for ext in annotated_extensions:
            image_extension = os.path.splitext(ext)[1]
            if image_extension.lower() in ['.jpg', '.tif', '.jpeg', '.tiff', '.png']:
                approved_ext_nt = True
        print(f'found approved ext NT for {patient}? ', str(approved_ext_nt))
        if approved_ext and approved_ext_ann and approved_ext_nt:
            # finding = df.loc[df['Number'] == patient_num]['Quality'].values
            # if finding.size > 0:
            #     finding = finding[0]
            #     print(f'patient {patient_num} quality = {finding}')
            #     if '4' in finding or '5' in finding:
            data.append(patient)
    except:
        print('skipping ' + patient)

print('total patients: ' + str(len(data)))

for patient_name in data:

    patient_path = os.path.join(DATA_FOLDER, patient_name)
    patient = patient_name.split(' ')[1]

    extensions = glob.glob(f'{patient_path}/{patient}.3_cropped.*')
    for ext in extensions:
        image_extension = os.path.splitext(ext)[1]
        if image_extension.lower() in ['.jpg', '.tif', '.jpeg', '.tiff', '.png']:
            cropped_image_extension = image_extension
    cropped_image = f'{patient_path}/{patient}.3_cropped{cropped_image_extension}'
    us_img = np.array((Image.open(cropped_image)).getchannel("R"))
    us_img = st.resize(us_img, us_img.shape)
    img_save = f'{IMAGE_FOLDER}/images/{patient}_3.jpg'
    cv2.imwrite(img_save, cv2.convertScaleAbs(us_img, alpha=(255.0)))

    extensions = glob.glob(f'{patient_path}/{patient}.3_cropped_headseg.*')
    for ext in extensions:
        image_extension = os.path.splitext(ext)[1]
        if image_extension.lower() in ['.jpg', '.tif', '.jpeg', '.tiff', '.png']:
            baby_annotated_file_extension = image_extension
    baby_path = f'{patient_path}/{patient}.3_cropped_headseg{baby_annotated_file_extension}'
    baby_mask = get_mask(baby_path)
    baby_mask[baby_mask > 0] = 255.0
    baby_save = f'{IMAGE_FOLDER}/masks/{patient}_3_baby_mask.jpg'
    cv2.imwrite(baby_save, cv2.convertScaleAbs(baby_mask, alpha=(255.0)))

    extensions = glob.glob(f'{patient_path}/{patient}.3_cropped_NT.*')
    NT_annotated_file_extension = ''
    for ext in extensions:
        image_extension = os.path.splitext(ext)[1]
        if image_extension.lower() in ['.jpg', '.tif', '.jpeg', '.tiff', '.png']:
            NT_annotated_file_extension = image_extension
    if NT_annotated_file_extension:
        NT_path = f'{patient_path}/{patient}.3_cropped_NT{NT_annotated_file_extension}'
        NT_mask = get_mask(NT_path, True)
        NT_mask[NT_mask > 0] = 255.0
        NT_save = f'{IMAGE_FOLDER}/masks/{patient}_3_nt_mask.jpg'
        cv2.imwrite(NT_save, cv2.convertScaleAbs(NT_mask, alpha=(255.0)))

    try:
        anomaly_extension = os.path.splitext(glob.glob(f'{patient_path}/{patient}.3 annotated_cropped.*')[0])[1]
        anomaly_path = f'{patient_path}/{patient}.3 annotated_cropped{anomaly_extension}'
        anomaly_mask = get_mask(anomaly_path)
        anomaly_mask[anomaly_mask>0] = 255.0
    except:
        try:
            anomaly_extension = os.path.splitext(glob.glob(f'{patient_path}/{patient}.3 marked_cropped.*')[0])[1]
            anomaly_path = f'{patient_path}/{patient}.3 marked_cropped{anomaly_extension}'
            anomaly_mask = get_mask(anomaly_path)
            anomaly_mask[anomaly_mask > 0] = 255.0
        except:
            anomaly_mask = np.zeros(us_img.shape)

    anomaly_save = f'{IMAGE_FOLDER}/masks/{patient}_3_anomaly_mask.jpg'
    cv2.imwrite(anomaly_save, cv2.convertScaleAbs(anomaly_mask, alpha=(255.0)))
    print(f'saved patient {patient} in test')
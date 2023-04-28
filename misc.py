import pandas as pd
import numpy as np
import skimage.transform as st
from pathlib import Path
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os


# LOSS_CSV = '/Users/bhavyakasera/Documents/temp/two_channel_nt/Two_Channel_Train_Loss.csv'
# df = pd.read_csv(LOSS_CSV)

# print(df['val_loss'].min(), df['val_loss'].argmin(), df['val_loss'][44])

# plt.plot(np.arange(len(df['train_loss'])), df['train_loss'], label='training loss')
# plt.plot(np.arange(len(df['val_loss'])), df['val_loss'], label='validation loss')
# plt.legend()
# plt.show()

# def dice_score(inputs, targets, smooth=1):
#     # flatten label and prediction tensors
#     # inputs = inputs.reshape(-1)
#     # targets = targets.reshape(-1)

#     intersection = (inputs * targets).sum()
#     dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

#     # print(f'Sum of Product of inputs & targets: {(inputs * targets).sum()}')
#     # print(f'Numerator: {(2. * intersection + smooth)} , Denominator: {(inputs.sum() + targets.sum() + smooth)}')

#     return dice


# DATA_FOLDER = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/data/PrenatalUS_two_channel_test'
# patients = os.listdir(f'{DATA_FOLDER}/images')
# total_dice = 0
# for patient in patients:
#     patient_num = patient.split('_')[0]
#     baby_ground = cv2.imread(f'{DATA_FOLDER}/masks/{patient_num}_3_baby_mask.jpg', cv2.IMREAD_GRAYSCALE)
#     baby_ground = cv2.resize(baby_ground, (512, 512), interpolation=cv2.INTER_AREA)
#     print(baby_ground[baby_ground>0].size)
#     nt_ground = cv2.imread(f'{DATA_FOLDER}/masks/{patient_num}_3_nt_mask.jpg', cv2.IMREAD_GRAYSCALE)
#     nt_ground = cv2.resize(nt_ground, (512, 512), interpolation=cv2.INTER_AREA)
#     baby_pred = cv2.imread(f'{DATA_FOLDER}/masks/{patient_num}_3_baby_mask_pred.jpg', cv2.IMREAD_GRAYSCALE)
#     nt_pred = cv2.imread(f'{DATA_FOLDER}/masks/{patient_num}_3_nt_mask_pred.jpg', cv2.IMREAD_GRAYSCALE)
#     preds = np.stack((baby_pred, nt_pred), axis=0)
#     ground = np.stack((baby_ground, nt_ground), axis=0)
#     total_dice += dice_score(preds, ground)
# print("num patients = ", len(patients))
# print("total dice = ", total_dice, "avg dice = ", total_dice/len(patients))

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
import matplotlib.pyplot as plt

DATA_FOLDER = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/data/prenatalUS_two_channel_test'
DATASHEET_PATH = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/prenatal_data_info.csv'
DEMO_PATH = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/DataDemographics_24.04.2023.csv'
# DICEFILE_PATH = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/results/two_channel_pretrained_test_dice.csv'
df = pd.read_csv(DATASHEET_PATH)
df_demo = pd.read_csv(DEMO_PATH)
train_normal = pd.DataFrame(columns=['quality', 'confidence', 'mat_age', 'gest_age', 'bmi', 'gravida', 'cosang'])
train_thick = pd.DataFrame(columns=['quality', 'confidence', 'mat_age', 'gest_age', 'bmi', 'gravida', 'cosang'])



patients = os.listdir(f'{DATA_FOLDER}/images')

for patient in patients:
    patient_num = patient.split('_')[0]
    finding = df.loc[df['Number']==patient_num]['Finding'].values
    if len(df.loc[df['Number']==patient_num]['Quality'].values) < 1:
        qual = 1
        conf = 1
    else:
        try:
            qual = int(df.loc[df['Number']==patient_num]['Quality'].values[0])
            conf = int(df.loc[df['Number']==patient_num]['Certainty'].values[0])
        except ValueError:
            qual = 1
            conf = 1
            
    patient_num = float(patient_num)
    if patient_num<560.0:
        mat_age = int(df_demo.loc[df_demo['Pt_id']==patient_num]['Maternal age (years)'].values[0])
        gest_age = int(df_demo.loc[df_demo['Pt_id']==patient_num]['Gestational age (days)'].values[0])
        bmi = float(df_demo.loc[df_demo['Pt_id']==patient_num]['Maternal BMI'].values[0])
        gravida = int(df_demo.loc[df_demo['Pt_id']==patient_num]['Gravida'].values[0])
        cosang = df_demo.loc[df_demo['Pt_id']==patient_num]['Cosanguinity'].values[0]
    else:
        mat_age = 18
        gest_age = 80
        bmi = 25
        gravida = 1
        cosang = 'no'

    if finding.size > 0:
        finding = finding[0]
        if isinstance(finding, str):
            finding = finding.lower()
            if 'thick' in finding and 'nt' in finding:
                train_thick.loc[len(train_thick)] = [qual, conf, mat_age, gest_age, bmi, gravida, cosang]
            else:
                train_normal.loc[len(train_normal)] = [qual, conf, mat_age, gest_age, bmi, gravida, cosang]
        else:
            train_normal.loc[len(train_normal)] = [qual, conf, mat_age, gest_age, bmi, gravida, cosang]
    else:
        train_normal.loc[len(train_normal)] = [qual, conf, mat_age, gest_age, bmi, gravida, cosang]

# patients = os.listdir(f'{DATA_FOLDER}_val/images')

# for patient in patients:
#     patient_num = patient.split('_')[0]
#     finding = df.loc[df['Number']==patient_num]['Finding'].values
#     if len(df.loc[df['Number']==patient_num]['Quality'].values) < 1:
#         qual = 1
#         conf = 1
#     else:
#         try:
#             qual = int(df.loc[df['Number']==patient_num]['Quality'].values[0])
#             conf = int(df.loc[df['Number']==patient_num]['Certainty'].values[0])
#         except ValueError:
#             qual = 1
#             conf = 1
            
#     patient_num = float(patient_num)
#     if patient_num<560.0:
#         mat_age = int(df_demo.loc[df_demo['Pt_id']==patient_num]['Maternal age (years)'].values[0])
#         gest_age = int(df_demo.loc[df_demo['Pt_id']==patient_num]['Gestational age (days)'].values[0])
#         bmi = float(df_demo.loc[df_demo['Pt_id']==patient_num]['Maternal BMI'].values[0])
#         gravida = int(df_demo.loc[df_demo['Pt_id']==patient_num]['Gravida'].values[0])
#         cosang = df_demo.loc[df_demo['Pt_id']==patient_num]['Cosanguinity'].values[0]
#     else:
#         mat_age = 1
#         gest_age = 1
#         bmi = 1
#         gravida = 1
#         cosang = 'no'

#     if finding.size > 0:
#         finding = finding[0]
#         if isinstance(finding, str):
#             finding = finding.lower()
#             if 'thick' in finding and 'nt' in finding:
#                 train_thick.loc[len(train_thick)] = [qual, conf, mat_age, gest_age, bmi, gravida, cosang]
#             else:
#                 train_normal.loc[len(train_normal)] = [qual, conf, mat_age, gest_age, bmi, gravida, cosang]
#         else:
#             train_normal.loc[len(train_normal)] = [qual, conf, mat_age, gest_age, bmi, gravida, cosang]
#     else:
#         train_normal.loc[len(train_normal)] = [qual, conf, mat_age, gest_age, bmi, gravida, cosang]

print('normal qual mean, std = ', train_normal['quality'].mean(), train_normal['quality'].std())
print('normal conf mean, std = ', train_normal['confidence'].mean(), train_normal['confidence'].std())
print('normal mat age mean, std = ', train_normal['mat_age'].mean(), train_normal['mat_age'].std())
print('normal gest age mean, std = ', train_normal['gest_age'].mean(), train_normal['gest_age'].std())
print('normal mat bmi mean, std = ', train_normal['bmi'].mean(), train_normal['bmi'].std())
print('normal gravida mean, std = ', train_normal['gravida'].mean(), train_normal['gravida'].std())
print('normal cosang percentage = ', len(train_normal.loc[train_normal['cosang']=='yes'])/len(train_normal))
print('thick qual mean, std = ', train_thick['quality'].mean(), train_thick['quality'].std())
print('thick conf mean, std = ', train_thick['confidence'].mean(), train_thick['confidence'].std())
print('thick mat age mean, std = ', train_thick['mat_age'].mean(), train_thick['mat_age'].std())
print('thick gest age mean, std = ', train_thick['gest_age'].mean(), train_thick['gest_age'].std())
print('thick mat bmi mean, std = ', train_thick['bmi'].mean(), train_thick['bmi'].std())
print('thick gravida mean, std = ', train_thick['gravida'].mean(), train_thick['gravida'].std())
print('thick cosang percentage = ', len(train_thick.loc[train_thick['cosang']=='yes'])/len(train_thick))

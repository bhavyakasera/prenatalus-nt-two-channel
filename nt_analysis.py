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

DATA_FOLDER = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/data/prenatalUS_two_channel_test_pretrained_qual'
DATASHEET_PATH = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/prenatal_data_info.csv'
DICEFILE_PATH = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/results/two_channel_pretrained_test_dice.csv'
df = pd.read_csv(DATASHEET_PATH)
df_dice = pd.read_csv(DICEFILE_PATH)

patients = os.listdir(f'{DATA_FOLDER}/images')
print('number of patients = ', len(patients))

colours = []
NT_to_baby = []
status = []
count_thick = 0

tp_low = 0
pred_pos_low = 0
pred_neg_low = 0
tn_low = 0

tp_high = 0
pred_pos_high = 0
pred_neg_high = 0
tn_high = 0

# tp = 0
# tn = 0
# pred_pos = 0
# pred_neg = 0

for patient in patients:
    patient_num = patient.split('_')[0]
    baby = cv2.imread(f'{DATA_FOLDER}/masks/{patient_num}_3_baby_mask_pred.jpg', cv2.IMREAD_GRAYSCALE)
    nt = cv2.imread(f'{DATA_FOLDER}/masks/{patient_num}_3_nt_mask_pred.jpg', cv2.IMREAD_GRAYSCALE)
    finding = df.loc[df['Number']==patient_num]['Finding'].values
    print(f'for patient {patient_num} NT to baby ratio = {(nt>0).sum()/(baby>0).sum()}, finding = {finding}')
    # dice = float(df_dice.loc[df_dice['patient_num']==(patient_num+'_3')]['dice_score'].values[0])

    ratio = (nt>0).sum()/(baby>0).sum()
    NT_to_baby.append(ratio)
    if finding.size > 0:
        finding = finding[0]
        if isinstance(finding, str):
            finding = finding.lower()
            if 'thick' in finding and 'nt' in finding:
                colours.append('red')
                status.append('infected')
                count_thick += 1
            else:
                colours.append('blue')
                status.append('uninfected')
        else:
            colours.append('blue')
            status.append('uninfected')
    else:
        colours.append('blue')
        status.append('uninfected')

    if len(df.loc[df['Number']==patient_num]['Quality'].values) < 1:
        qual = 1
    else:
        qual = int(df.loc[df['Number']==patient_num]['Quality'].values[0])
    high_qual = False
    if qual >= 4:
        high_qual = True
    
    if ratio > 0.0385:
        pred = 'infected'
        # if high_qual:
        #     pred_pos_high += 1
        # else:
        #     pred_pos_low += 1
        if high_qual:
            pred_pos_high += 1
        else:
            pred_pos_low += 1 
    else:
        pred = 'uninfected'
        # if high_qual:
        #     pred_neg_high += 1
        # else:
        #     pred_neg_low += 1
        if high_qual:
            pred_neg_high += 1
        else:
            pred_neg_low += 1
    
    if status[-1]=='infected' and pred == 'infected':
        if high_qual:
            tp_high += 1
        else:
            tp_low += 1
        # tp += 1
    elif status[-1]=='uninfected' and pred=='uninfected':
        if high_qual:
            tn_high += 1
        else:
            tn_low += 1
        # tn += 1

    # if ratio > 0.043:
    #     colours.append('red')
    #     status.append('infected')
    # else:
    #     colours.append('blue')
    #     status.append('uninfected')


    # if ((anomaly&nt) > 0).sum() > 0:
    #     colours.append('red')
    # else:
    #     colours.append('blue')

# patients = os.listdir(f'{DATA_FOLDER}_val/images')
#
# for patient in patients:
#     patient_num = patient.split('_')[0]
#     baby = cv2.imread(f'{DATA_FOLDER}_val/masks/{patient_num}_3_baby_mask.jpg', cv2.IMREAD_GRAYSCALE)
#     nt = cv2.imread(f'{DATA_FOLDER}_val/masks/{patient_num}_3_NT_mask.jpg', cv2.IMREAD_GRAYSCALE)
#     # anomaly = cv2.imread(f'{DATA_FOLDER}_val/masks/{patient_num}_3_anomaly_mask.jpg', cv2.IMREAD_GRAYSCALE)
#     # print(f'for patient {patient_num} NT to baby ratio = {(nt>0).sum()/(baby>0).sum()}, anomaly size = {((anomaly&nt)>0).sum()}')
#
#     finding = df.loc[df['Number'] == patient_num]['Finding'].values
#     print(f'for patient {patient_num} NT to baby ratio = {(nt > 0).sum() / (baby > 0).sum()}, finding = {finding}')
#
#     NT_to_baby.append((nt > 0).sum() / (baby > 0).sum())
#     if finding.size > 0:
#         finding = finding[0]
#         if isinstance(finding, str):
#             finding = finding.lower()
#             if 'thick' in finding and 'nt' in finding:
#                 colours.append('red')
#                 status.append('infected')
#                 count_thick += 1
#             else:
#                 colours.append('blue')
#                 status.append('uninfected')
#         else:
#             print(type(finding))
#             colours.append('blue')
#             status.append('uninfected')
#     else:
#         colours.append('blue')
#         status.append('uninfected')
#
#     # if ((anomaly&nt) > 0).sum() > 0:
#     #     colours.append('red')
#     # else:
#     #     colours.append('blue')

df_save = pd.DataFrame(data={'Status':status, 'Ratio':NT_to_baby})
# df_save.to_csv('C:/Users/Bhavya Kasera/Documents/PrenatalUS/results/test_two_channel_analysis.csv')
x = np.arange(0, len(colours))
# print("total positive =", count_thick)
# print("total negative =", 109-count_thick)
# print("true positive =", tp)
# print("true negative =", tn)
# fp = pred_pos-tp
# fn = pred_neg-tn
# print("total predicted positive =", pred_pos)
# print("total predicted negative =", pred_neg)
# print("PPV high =", tp_high/pred_pos_high)
# print("NPV high =", tn_high/pred_neg_high)
# print("PPV low =", tp_low/pred_pos_low)
# print("NPV low =", tn_low/pred_neg_low)
ppv_high = tp_high/pred_pos_high
npv_high = tn_high/pred_neg_high
print("PPV =", ppv_high)
print("NPV =", npv_high)
se_ppv_high = ((ppv_high*(1-ppv_high))/(pred_pos_high))**0.5
se_npv_high = ((npv_high*(1-npv_high))/(pred_neg_high))**0.5
print("SE PPV high = ", se_ppv_high)
print("SE NPV high = ", se_npv_high)
print("95% CI PPV high = +-", 1.96*se_ppv_high)
print("95% CI PPV high = +-", 1.96*se_npv_high)
ppv_low = tp_low/pred_pos_low
npv_low = tn_low/pred_neg_low
print("PPV low =", ppv_low)
print("NPV low =", npv_low)
se_ppv_low = ((ppv_low*(1-ppv_low))/(pred_pos_low))**0.5
se_npv_low = ((npv_low*(1-npv_low))/(pred_neg_low))**0.5
print("SE PPV low = ", se_ppv_low)
print("SE NPV low = ", se_npv_low)
print("95% CI PPV low = +-", 1.96*se_ppv_low)
print("95% CI PPV low = +-", 1.96*se_npv_low)
# plt.figure()
# plt.scatter(x, NT_to_baby, c=colours)
# plt.legend(['normal NT', 'thick NT'])
# plt.title('Ratio of NT size to baby size')
# plt.show()
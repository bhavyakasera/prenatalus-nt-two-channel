import glob
import os
import numpy as np
import os
import pathlib

import pandas as pd
import numpy as np
from pathlib import Path

DATASHEET_PATH = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/prenatal_data_info.csv'
DICEFILE_PATH_FP = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/results/two_channel_pretrained_test_dice.csv'
DICEFILE_PATH_QP = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/results/two_channel_test_pretrained_qual_dice.csv'
DICEFILE_PATH_FNP = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/results/two_channel_test_dice.csv'
DICEFILE_PATH_QNP = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/results/two_channel_test_qual_dice.csv'

df = pd.read_csv(DATASHEET_PATH)
df_dice_fp = pd.read_csv(DICEFILE_PATH_FP)
df_dice_qp = pd.read_csv(DICEFILE_PATH_QP)
df_dice_fnp = pd.read_csv(DICEFILE_PATH_FNP)
df_dice_qnp = pd.read_csv(DICEFILE_PATH_QNP)

print("full pretrain 95% CI = ", df_dice_fp['dice_score'].std()*2)
print("qual pretrain 95% CI = ", df_dice_qp['dice_score'].std()*2)
print("full no pretrain 95% CI = ", df_dice_fnp['dice_score'].std()*2)
print("qual no pretrain 95% CI = ", df_dice_qnp['dice_score'].std()*2)

dice_high = []
patients_high = []
dice_low = []
patients_low = []

for i in range(len(df_dice_fp)):
    print(df_dice_fp.iloc[i, 1])
    patient_num = df_dice_fp.iloc[i, 1].split('_')[0]
    quality = df.loc[df['Number']==patient_num]['Quality'].values
    if len(quality) < 1:
        qual = 1
    else:
        qual = int(quality[0])
    if qual >= 4:
        dice_high.append(df_dice_fp.iloc[i,2])
        patients_high.append(df_dice_fp.iloc[i,1])
    else:
        dice_low.append(df_dice_fp.iloc[i,2])
        patients_low.append(df_dice_fp.iloc[i,1])

df_save_high = pd.DataFrame(data={'patient_num':patients_high, 'dice_score':dice_high})
df_save_low = pd.DataFrame(data={'patient_num':patients_low, 'dice_score':dice_low})
print('high 2*sd = ', df_save_high['dice_score'].std()*2)
print('low 2*sd = ', df_save_low['dice_score'].std()*2)
df_save_high.to_csv('C:/Users/Bhavya Kasera/Documents/PrenatalUS/results/test_two_channel_high_qual_dice.csv')
df_save_low.to_csv('C:/Users/Bhavya Kasera/Documents/PrenatalUS/results/test_two_channel_low_qual_dice.csv')
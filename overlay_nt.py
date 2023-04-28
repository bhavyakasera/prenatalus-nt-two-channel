from pathlib import Path
import cv2
import os

DATA_FOLDER = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/data/prenatalUS_two_channel_test_pretrained_qual/'
SAVE_FOLDER = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/results/test_nt_overlays_two_channel_qual_pretrained'

IMAGE_WIDTH, IMAGE_HEIGHT = (512, 512)

patients = os.listdir(f'{DATA_FOLDER}images')

for patient in patients:
    patient_num = patient.split('_')[0]

    main = cv2.imread(f'{DATA_FOLDER}images/{patient}', cv2.IMREAD_GRAYSCALE)
    main = cv2.cvtColor(main, cv2.COLOR_GRAY2BGR)
    main = cv2.resize(main, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{SAVE_FOLDER}/{patient_num}_3_resized.png', main)

    seg = cv2.imread(f'{DATA_FOLDER}masks/{patient_num}_3_nt_mask_pred.jpg', cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    line_thickness = 2
    cv2.drawContours(main, contours, -1, (0, 255, 255), line_thickness)

    # baby_seg = cv2.imread(f'{DATA_FOLDER}masks/{patient_num}_3_baby_mask_pred.jpg', cv2.IMREAD_GRAYSCALE)
    # baby_contours, _ = cv2.findContours(baby_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # cv2.drawContours(main, baby_contours, -1, (0, 0, 255), line_thickness)
    seg_ground = cv2.imread(f'{DATA_FOLDER}masks/{patient_num}_3_nt_mask.jpg', cv2.IMREAD_GRAYSCALE)
    seg_ground = cv2.resize(seg_ground, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_AREA)
    contours_ground, _ = cv2.findContours(seg_ground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(main, contours_ground, -1, (0, 255, 0), line_thickness)

    cv2.imwrite(f'{SAVE_FOLDER}/{patient_num}_3_nt_mask-overlay.png', main)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from segmentation_model import UNET, UNETResNet18

import cv2

from utils import (load_checkpoint, save_checkpoint, get_loaders_two_channel, generate_and_save_two_channel_masks)
from loss_functions import (DiceLoss, TwoChannelDiceLoss, DiceBCELoss, IoULoss, WeightedSoftDiceLoss)

# TODO: modify
TRAIN_TWO_CHANNEL = False
TEST_TWO_CHANNEL = True

# TODO: Modify Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 45
NUM_WORKERS = 2
IMAGE_HEIGHT = 512  # 1280 originally
IMAGE_WIDTH = 512  # 1918 originally
PIN_MEMORY = True
LMDA = 0

# TODO: Views and number of negative examples to include
VIEWS = [3]
NUMNEG = 45

# Depending on which view we are working with, the appropriate body segmentation is used.
if 2 in VIEWS:
    body_part = 'body'
elif 3 in VIEWS:
    body_part = 'head'

# TODO: Main output folder to save images to
OUTPUT_FOLDER = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/results/test_results_50_one_stage/'
DATASHEET_PATH = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/prenatal_data_info.csv'

# Makes subfolders for the different tasks within the main output folder

if TRAIN_TWO_CHANNEL:
    MODEL_FOLDER = f'C:/Users/Bhavya Kasera/Documents/PrenatalUS/models/two_channel_nt/'
    CSV_FOLDER = f'C:/Users/Bhavya Kasera/Documents/PrenatalUS/models/two_channel_nt/'

TEST_TWO_CHANNEL_LOAD_MODEL = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/models/two-channel-model-weights-qual-pretrained/nt_checkpoint.pth.tar'
TEST_IMG_DIR = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/data/prenatalUS_two_channel_test/images'
TEST_MASK_DIR = 'C:/Users/Bhavya Kasera/Documents/PrenatalUS/data/prenatalUS_two_channel_test/masks'
TEST_NAME = 'TwoChannelTest'
 
if TRAIN_TWO_CHANNEL:
    SAVE_MODEL_PATH = f'{MODEL_FOLDER}two-channel-model-weights/nt_checkpoint.pth.tar'

def two_channel_train_fn(loader, model, optimizer, loss_fn, scaler, folder, train_image_dice, epoch):
    df = pd.read_csv(DATASHEET_PATH)
    train_epoch_loss = []

    loop = tqdm(loader)
    for batch_idx, (data, targets, fnames) in enumerate(loop):

        # TODO: Mask the baby so as to block the rest of the ultrasound
        data = data.float().to(device=DEVICE)
        targets = targets.float().squeeze(2).to(device=DEVICE)

        #quality_list = []
        #for i in range(len(fnames)):
        #    patient_num = fnames[i].split('_')[0]
        #    quality = df.loc[df['Number']==patient_num]['Quality'].values
        #    print('quality = ', quality)
        #    if len(quality)==0:
        #        quality = [1]
        #    quality = quality[0]
        #    if not isinstance(quality, str):
        #        quality = 1
        #    quality_list.append(int(quality)/5)
        #print('two channel loss quality list = ', quality_list)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)

            # preds_anomaly, preds_baby = predictions.split(1, dim=1)
            # y_anomaly, y_baby = targets.squeeze(2).split(1, dim=1)

            # CHANGE FOR DICE
            loss = loss_fn(predictions, targets)
            print('Batch {} Two Channel Loss: {}'.format(batch_idx, loss))

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Save train epoch loss
        train_epoch_loss.append(loss.item())

        # Save per image dice score
        # for i in range(len(image_dice_lst)):
        #     if fnames[i] not in train_image_dice:
        #         train_image_dice[fnames[i]] = []
        #     train_image_dice[fnames[i]].append(image_dice_lst[i])

        preds_baby, preds_nt = predictions.split(1, dim=1)
        y_baby, y_nt = targets.squeeze(2).split(1, dim=1)

        # Note to self: Another instance of my super hacky hacky way of saving the images. Based on the assumption that
        # batch size is 2. Change this for the future.
        flag = 0
        try:
            preds_baby_one, preds_baby_two = preds_baby.split(1, dim=0)
            y_baby_one, y_baby_two = y_baby.split(1, dim=0)
            preds_nt_one, preds_nt_two = preds_nt.split(1, dim=0)
            y_nt_one, y_nt_two = y_nt.split(1, dim=0)
            flag = 1

        except ValueError:
            preds_baby_one = preds_baby
            y_baby_one = y_baby
            preds_nt_one = preds_nt
            y_nt_one = y_nt

        # preds_baby_one_df = pd.DataFrame(F.sigmoid(preds_baby_one).detach().cpu().squeeze().numpy())  # convert to a dataframe
        # preds_baby_one_df.to_csv(f"{CSV_FOLDER}/TRAIN-{fnames[0]}", index=False)  # save to file

        torchvision.utils.save_image(preds_baby_one, f"{folder}/TRAIN-baby-pred_{fnames[0]}.png")
        torchvision.utils.save_image(y_baby_one, f"{folder}/TRAIN-baby-{fnames[0]}.png")

        torchvision.utils.save_image(preds_nt_one, f"{folder}/TRAIN-NT-pred_{fnames[0]}.png")
        torchvision.utils.save_image(y_nt_one, f"{folder}/TRAIN-NT-{fnames[0]}.png")
        if '636_3' in fnames:
            torchvision.utils.save_image(preds_nt_one, f"{folder}/epochs/{epoch}_TRAIN-NT-pred_{fnames[0]}.png")
            torchvision.utils.save_image(preds_baby_one, f"{folder}/epochs/{epoch}_TRAIN-baby-pred_{fnames[0]}.png")

        if flag == 1:
            # preds_anomaly_two_df = pd.DataFrame(F.sigmoid(preds_anomaly_two).detach().cpu().squeeze().numpy())  # convert to a dataframe
            # preds_anomaly_two_df.to_csv(f"{CSV_FOLDER}/TRAIN-{fnames[1]}", index=False)  # save to file

            torchvision.utils.save_image(preds_baby_two, f"{folder}/TRAIN-baby-pred_{fnames[1]}.png")
            torchvision.utils.save_image(y_baby_two, f"{folder}/TRAIN-baby-{fnames[1]}.png")

            torchvision.utils.save_image(preds_nt_two, f"{folder}/TRAIN-NT-pred_{fnames[1]}.png")
            torchvision.utils.save_image(y_nt_two, f"{folder}/TRAIN-NT-{fnames[1]}.png")

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    # CHANGE FOR DICE
    return np.mean(np.array(train_epoch_loss))

def main():

    image_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.ToGray(),
            A.HorizontalFlip(p=0.5),  
            ToTensorV2(),
        ],
    )

    mask_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2(),
        ],
    )
    print("DEVICE = ", DEVICE)

    # Initialize the model (UNETResNet18 for pretrained, UNET otherwise)
    two_channel_model = UNETResNet18(out_channels=2).to(DEVICE)
    # two_channel_model = UNET(out_channels=2)

    # Initialize loss function
    # two_channel_loss_fn = nn.CrossEntropyLoss()
    two_channel_loss_fn = DiceLoss()

    # Initialize optimizer
    two_channel_optimizer = optim.Adam(two_channel_model.parameters(), lr=LEARNING_RATE)

    # Initialize grad scalar
    two_channel_scaler = torch.cuda.amp.GradScaler()

    # Image directory paths
    TRAIN_IMG_DIR = f'C:/Users/Bhavya Kasera/Documents/PrenatalUS/data/prenatalUS_NT_train/images'
    TRAIN_MASK_DIR = f'C:/Users/Bhavya Kasera/Documents/PrenatalUS/prenatalUS_NT_train/masks'
    VAL_IMG_DIR = f'C:/Users/Bhavya Kasera/Documents/PrenatalUS/data/prenatalUS_NT_val/images'
    VAL_MASK_DIR = f'C:/Users/Bhavya Kasera/Documents/PrenatalUS/data/prenatalUS_NT_val/masks'

    # Get data loaders
    train_loader, val_loader, test_loader = get_loaders_two_channel(
            TRAIN_IMG_DIR,
            TRAIN_MASK_DIR,
            VAL_IMG_DIR,
            VAL_MASK_DIR,
            TEST_IMG_DIR,
            TEST_MASK_DIR,
            BATCH_SIZE,
            image_transforms,
            image_transforms,
            mask_transforms,
            NUM_WORKERS,
            PIN_MEMORY
        )
    if TRAIN_TWO_CHANNEL:
        print("Training two-channel model")
    else:
        print("Testing two-channel model")

    # ------------------------------------------------------------------------------------------------------------------

    # Train a UNet for baby & NT prediction
    if TRAIN_TWO_CHANNEL:
        print("Training two channel model...")

        train_mean_loss = []
        val_mean_loss = []

        train_image_dice = {'Epoch': []}
        val_image_dice = {'Epoch': []}

        for epoch in range(NUM_EPOCHS):
            print(f"\nTRAIN Epoch: {epoch}" + 25*'*')

            # Set model to training mode
            two_channel_model.train()

            train_image_dice['Epoch'].append(epoch)

            # CHANGE FOR DICE
            epoch_train_loss = two_channel_train_fn(train_loader, two_channel_model, two_channel_optimizer, two_channel_loss_fn, two_channel_scaler, MODEL_FOLDER, train_image_dice=train_image_dice, epoch=epoch)
            train_mean_loss.append(epoch_train_loss)

            # save two_channel_model
            checkpoint = {
                "state_dict": two_channel_model.state_dict(),
                "two_channel_optimizer": two_channel_optimizer.state_dict()
            }
            save_checkpoint(checkpoint, SAVE_MODEL_PATH)

            # check accuracy
            # dice_score = check_accuracy(val_loader, anomaly_model, device=DEVICE)
            # dice_scores.append(dice_score.cpu())

            print(f"VALIDATION Epoch: {epoch}" + 25 * '*')

            # Set model to evaluation mode
            two_channel_model.eval()

            val_epoch_loss = []
            loop = tqdm(val_loader)
            val_image_dice['Epoch'].append(epoch)
            df = pd.read_csv(DATASHEET_PATH)

            for batch_idx, (data, targets, fnames) in enumerate(loop):
            #    quality_list = []
            #    for i in range(len(fnames)):
            #        patient_num = fnames[i].split('_')[0]
            #        quality = df.loc[df['Number']==patient_num]['Quality'].values
            #        print('quality = ', quality)
            #        if len(quality)==0:
            #            quality = [1]
            #        quality = quality[0]
            #        if not isinstance(quality, str):
            #            quality = 1
            #        quality_list.append(int(quality)/5)
                # TODO: Mask the baby so as to block the rest of the ultrasound
                data = data.float().to(device=DEVICE)
                targets = targets.float().squeeze(2).to(device=DEVICE)

                # forward
                with torch.no_grad():
                    predictions = two_channel_model(data)

                    # preds_baby, preds_nt = predictions.split(1, dim=1)
                    # y_baby, y_nt = targets.squeeze(2).split(1, dim=1)

                    # CHANGE FOR DICE
                    loss = two_channel_loss_fn(predictions, targets)
                    print('VALIDATION: Batch {} Two Channel Loss: {}'.format(batch_idx, loss))

                # Get the batch validation loss
                val_epoch_loss.append(loss.item())

                # Get each image dice score
                # for i in range(len(image_dice_lst)):
                #     if fnames[i] not in val_image_dice:
                #         val_image_dice[fnames[i]] = []
                #     val_image_dice[fnames[i]].append(image_dice_lst[i])

                # Save the predictions
                # Split the baby and anomaly predictions
                preds_baby, preds_nt = predictions.split(1, dim=1)
                y_baby, y_nt = targets.squeeze(2).split(1, dim=1)

                flag = 0
                try:
                    preds_baby_one, preds_baby_two = preds_baby.split(1, dim=0)
                    y_baby_one, y_baby_two = y_baby.split(1, dim=0)
                    preds_nt_one, preds_nt_two = preds_nt.split(1, dim=0)
                    y_nt_one, y_nt_two = y_nt.split(1, dim=0)
                    flag = 1

                except ValueError:
                    preds_baby_one = preds_baby
                    y_baby_one = y_baby
                    preds_nt_one = preds_nt
                    y_nt_one = y_nt

                # preds_baby_one_df = pd.DataFrame(F.sigmoid(preds_baby_one).detach().cpu().squeeze().numpy())  # convert to a dataframe
                # preds_baby_one_df.to_csv(f"{CSV_FOLDER}/TRAIN-{fnames[0]}", index=False)  # save to file

                torchvision.utils.save_image(preds_baby_one, f"{MODEL_FOLDER}/VAL-baby-pred_{fnames[0]}.png")
                torchvision.utils.save_image(y_baby_one, f"{MODEL_FOLDER}/VAL-baby-{fnames[0]}.png")

                torchvision.utils.save_image(preds_nt_one, f"{MODEL_FOLDER}/VAL-NT-pred_{fnames[0]}.png")
                torchvision.utils.save_image(y_nt_one, f"{MODEL_FOLDER}/VAL-NT-{fnames[0]}.png")

                if flag == 1:
                    # preds_anomaly_two_df = pd.DataFrame(F.sigmoid(preds_anomaly_two).detach().cpu().squeeze().numpy())  # convert to a dataframe
                    # preds_anomaly_two_df.to_csv(f"{CSV_FOLDER}/TRAIN-{fnames[1]}", index=False)  # save to file

                    torchvision.utils.save_image(preds_baby_two, f"{MODEL_FOLDER}/VAL-baby-pred_{fnames[1]}.png")
                    torchvision.utils.save_image(y_baby_two, f"{MODEL_FOLDER}/VAL-baby-{fnames[1]}.png")

                    torchvision.utils.save_image(preds_nt_two, f"{MODEL_FOLDER}/VAL-NT-pred_{fnames[1]}.png")
                    torchvision.utils.save_image(y_nt_two, f"{MODEL_FOLDER}/VAL-NT-{fnames[1]}.png")

                # update tqdm loop
                loop.set_postfix(loss=loss.item())

            # Get the average validation loss for the epoch_1
            val_mean_loss.append(np.mean(np.array(val_epoch_loss)))

        loss_df = pd.DataFrame(
            {"train_loss": train_mean_loss, "val_loss": val_mean_loss})  # move this into training_loop function
        loss_csvfile = MODEL_FOLDER + "Two_Channel_Train_Loss.csv"

        # train_image_dice_df = pd.DataFrame(train_image_dice)
        # train_img_dice_csvfile = MODEL_FOLDER + "Train_Image_Dice.csv"

        # print(val_image_dice)
        # val_image_dice_df = pd.DataFrame(val_image_dice)
        # val_img_dice_csvfile = MODEL_FOLDER + "Val_Image_Dice.csv"

        loss_df.to_csv(loss_csvfile)
        # train_image_dice_df.to_csv(train_img_dice_csvfile)
        # val_image_dice_df.to_csv(val_img_dice_csvfile)

    # ------------------------------------------------------------------------------------------------------------------

    if TEST_TWO_CHANNEL:
        print("Testing two channel...")

        # Load the trained anomaly model
        load_checkpoint(torch.load(TEST_TWO_CHANNEL_LOAD_MODEL, map_location=DEVICE), two_channel_model)
        two_channel_model.eval()

        test_batch_loss = []
        test_dice = []
        patient_nums = []
        img_dice_total = 0
        img_count = 0

        loop = tqdm(test_loader)

        for batch_idx, (data, targets, fnames) in enumerate(loop):
            print(f"Fnames: {fnames}")

            # data = mask_over(data, fnames, anomaly_folder.replace('anomaly/', 'masks/'))
            data = data.float().to(device=DEVICE)
            # targets = targets.float().squeeze(2).to(device=DEVICE)

            with torch.no_grad():
                predictions = two_channel_model(data)

                # print('TEST: Batch {} Anomaly Loss: {}'.format(batch_idx, loss))

                # Save the predictions
                # Split the baby and anomaly predictions
                preds_baby, preds_nt = predictions.split(1, dim=1)
                y_baby, y_nt = targets.squeeze(2).split(1, dim=1)

                # CHANGE FOR DICE
                loss, image_dice_lst = two_channel_loss_fn(predictions, targets)
                print('TEST: Batch {} two channel loss: {}'.format(batch_idx, loss))

                # Get the batch validation loss
                test_batch_loss.append(loss.item())

                img_dice_total += sum(image_dice_lst)
                img_count += len(image_dice_lst)

                flag = 0
                try:
                    preds_baby_one, preds_baby_two = preds_baby.split(1, dim=0)
                    y_baby_one, y_baby_two = y_baby.split(1, dim=0)
                    preds_nt_one, preds_nt_two = preds_nt.split(1, dim=0)
                    y_nt_one, y_nt_two = y_nt.split(1, dim=0)
                    flag = 1
                    

                except ValueError:
                    preds_baby_one = preds_baby
                    y_baby_one = y_baby
                    preds_nt_one = preds_nt
                    y_nt_one = y_nt

                # torchvision.utils.save_image(preds_baby_one,
                #                              f"{TEST_MASK_DIR}/{fnames[0]}_baby_mask_pred.jpg")
                # torchvision.utils.save_image(preds_nt_one,
                #                              f"{TEST_MASK_DIR}/{fnames[0]}_nt_mask_pred.jpg")
                test_dice.append(image_dice_lst[0])
                patient_nums.append(fnames[0])

                if flag == 1:
                    # torchvision.utils.save_image(preds_baby_two,
                    #                              f"{TEST_MASK_DIR}/{fnames[1]}_baby_mask_pred.jpg")
                    # torchvision.utils.save_image(preds_nt_two,
                    #                              f"{TEST_MASK_DIR}/{fnames[1]}_nt_mask_pred.jpg")
                    test_dice.append(image_dice_lst[1])
                    patient_nums.append(fnames[1])

                # update tqdm loop
                loop.set_postfix(loss=loss.item())

        # Get the average test loss for the test set
        avg_test_loss = np.mean(np.array(test_batch_loss))
        print(f"Average test loss: {avg_test_loss}")
        print(f"avg image dice:", img_dice_total/img_count, "total images = ", img_count)
        dice_df = pd.DataFrame(
            {"patient_num": patient_nums, "dice_score": test_dice})  # move this into training_loop function
        dice_csvfile = "C:/Users/Bhavya Kasera/Documents/PrenatalUS/results/two_channel_test_pretrained_qual_dice.csv"
        dice_df.to_csv(dice_csvfile)
        # load_checkpoint(torch.load(TEST_TWO_CHANNEL_LOAD_MODEL, map_location=DEVICE), two_channel_model)
        # test_loader = get_baby_mask_loader(TEST_IMG_DIR, TEST_MASK_DIR, BATCH_SIZE, image_transforms, mask_transforms, NUM_WORKERS, PIN_MEMORY)
        # generate_and_save_two_channel_masks(test_loader, two_channel_model, TEST_MASK_DIR, DEVICE)

if __name__ == "__main__":
    main()

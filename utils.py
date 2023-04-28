import numpy as np
import torch
import torchvision
from dataset import PreNatalTwoChannelDataset, PreNatalSegDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])




def get_loaders_two_channel(train_dir, train_maskdir, val_dir, val_maskdir, test_dir, test_maskdir, batch_size, train_transform, val_transform, mask_transform, num_workers=4, pin_memory=True):
    train_ds = PreNatalTwoChannelDataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform, mask_transform=mask_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    val_ds = PreNatalTwoChannelDataset(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform, mask_transform=mask_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    test_ds = PreNatalTwoChannelDataset(image_dir=test_dir, mask_dir=test_maskdir, transform=val_transform, mask_transform=mask_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader, test_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    batch_dice_scores = []
    model.eval()

    with torch.no_grad():
        for x, y, fnames in loader:
            x = x.float().to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()  # Converts any num > 0.5 to 1.0 and < 0.5 to 0.0

            # print(y)
            # print(y.shape)
            # print('Data Type: %s' % y.dtype)
            # print('Min: %.3f, Max: %.3f' % (y.min(), y.max()))
            y /= 255.0
            # print('Min: %.3f, Max: %.3f' % (y.min(), y.max()))
            # assert (y >= 0).all() and (y <= 255).all()
            # assert (preds >= 0).all() and (preds <= 1).all()
            # exit(99)

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            batch_dice_score = (2 * (preds * y).sum()) / (  # Summing the number of pixels where they are both the same (i.e. outputting a white pixel)
                (preds + y).sum() + 1e-3  # Summing the number of white pixels for preds and label
            )
            dice_score += batch_dice_score
            batch_dice_scores.append(batch_dice_score)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Average Dice score: {dice_score/len(loader)}")
    print(f"Batch dice scores: {str(batch_dice_scores)}")

    model.train()
    return dice_score/len(loader)


def assess_model(loader, model, device="cuda", model_type = ''):
    print(f'Performance for {model_type}')
    sensitivity = 0
    specificity = 0
    num_positive = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for x, y, fnames in loader:
            x = x.float().to(device)

            preds = torch.sigmoid(model(x))
            preds = preds>0.5

            preds_anomaly, preds_baby = preds.split(1, dim=1)
            y_anomaly, y_baby = y.squeeze(2).split(1, dim=1)
            flag = 0
            try:
                y_anomaly1, y_anomaly2 = y_anomaly.squeeze(2).split(1, dim=0)
                y_anomaly1, y_anomaly2 = (y_anomaly1>0), (y_anomaly2>0)
                preds_anomaly1, preds_anomaly2 = preds_anomaly.squeeze(2).split(1, dim=0)
                flag = 1
            except ValueError:
                y_anomaly1 = y_anomaly
                y_anomaly1 = y_anomaly1>0
                preds_anomaly1 = preds_anomaly

            if y_anomaly1.sum() > 0:
                sensitivity += (preds_anomaly1&y_anomaly1).sum()/y_anomaly1.sum()
                num_positive += 1
            specificity += ((~y_anomaly1)&(~preds_anomaly1)).sum()/(~y_anomaly1).sum()
            count += 1
            if flag:
                if y_anomaly2.sum() > 0:
                    sensitivity += (preds_anomaly2 & y_anomaly2).sum() / y_anomaly2.sum()
                    num_positive += 1
                specificity += ((~y_anomaly2) & (~preds_anomaly2)).sum() / (~y_anomaly2).sum()
                count += 1

    print("Avg sensitivity (correct positive/total positive) = ", (sensitivity/num_positive).item())
    print("Avg specificity (correct negative/total negative) = ", (specificity/count).item())



def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda", train_baby=False, train_anomaly=True):

    model.eval()

    for idx, (x, y, fnames) in enumerate(loader):
        x = x.float().to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        preds_anomaly, preds_baby = preds.split(1, dim=1)
        y_anomaly, y_baby = y.squeeze(2).split(1, dim=1)

        print(preds_anomaly.shape, preds_baby.shape, y_anomaly.shape, y_baby.shape)

        if train_baby:
            flag = 0
            try:
                preds_baby_one, preds_baby_two = preds_baby.split(1, dim=0)
                y_baby_one, y_baby_two = y_baby.split(1, dim=0)
                flag = 1

            except ValueError:
                preds_baby_one = preds_baby
                y_baby_one = y_baby

            torchvision.utils.save_image(preds_baby_one, f"{folder}/pred-baby_{fnames[0]}.png")
            torchvision.utils.save_image(y_baby_one, f"{folder}/baby_{fnames[0]}.png")

            if flag == 1:
                torchvision.utils.save_image(preds_baby_two, f"{folder}/pred-baby_{fnames[1]}.png")
                torchvision.utils.save_image(y_baby_two, f"{folder}/baby_{fnames[1]}.png")

        if train_anomaly:
            flag = 0
            try:
                preds_anomaly_one, preds_anomaly_two = preds_anomaly.split(1, dim=0)
                y_anomaly_one, y_anomaly_two = y_anomaly.split(1, dim=0)
                flag = 1

            except ValueError:
                preds_anomaly_one = preds_anomaly
                y_anomaly_one = y_anomaly

            torchvision.utils.save_image(preds_anomaly_one, f"{folder}/pred-anomaly_{fnames[0]}.png")
            torchvision.utils.save_image(y_anomaly_one, f"{folder}/anomaly_{fnames[0]}.png")

            if flag == 1:
                torchvision.utils.save_image(preds_anomaly_two, f"{folder}/pred-anomaly_{fnames[1]}.png")
                torchvision.utils.save_image(y_anomaly_two, f"{folder}/anomaly_{fnames[1]}.png")

    model.train()


def generate_and_save_seg_masks(loader, model, folder, seg_part='baby', device='cuda'):
    model.eval()

    for idx, (x, y, fnames) in enumerate(loader):
        print(f'generating {seg_part} masks, fnames = ', fnames)
        x = x.float().to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        flag = 0
        try:
            preds_seg_one, preds_seg_two = preds.split(1, dim=0)
            flag = 1

        except ValueError:
            preds_seg_one = preds

        torchvision.utils.save_image(preds_seg_one, f"{folder}/{fnames[0]}_{seg_part}_mask_pred.jpg")
        if flag == 1:
            torchvision.utils.save_image(preds_seg_two, f"{folder}/{fnames[1]}_{seg_part}_mask_pred.jpg")


def generate_and_save_two_channel_masks(loader, model, folder, device='cuda'):
    model.eval()

    for idx, (x, y, fnames) in enumerate(loader):
        print(f'generating two channel masks, fnames = ', fnames)
        x = x.float().to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        preds_baby, preds_nt = preds.split(1, dim=1)

        flag = 0
        try:
            preds_baby_one, preds_baby_two = preds_baby.split(1, dim=0)
            preds_nt_one, preds_nt_two = preds_nt.split(1, dim=0)
            flag = 1

        except ValueError:
            preds_baby_one = preds_baby
            preds_nt_one = preds_nt

        torchvision.utils.save_image(preds_baby_one, f"{folder}/{fnames[0]}_baby_mask.jpg")
        torchvision.utils.save_image(preds_baby_one, f"{folder}/{fnames[0]}_nt_mask.jpg")
        if flag == 1:
            torchvision.utils.save_image(preds_baby_two, f"{folder}/{fnames[1]}_baby_mask.jpg")
            torchvision.utils.save_image(preds_baby_two, f"{folder}/{fnames[1]}_nt_mask.jpg")




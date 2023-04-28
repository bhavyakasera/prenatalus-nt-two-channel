import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, baby_mask=None, smooth=1, lmda=0.9):
        image_dice_lst = []
        input_copy = F.sigmoid(inputs)
        target_copy = targets

        # comment out if your model contains a sigmoid or equivalent activation layer
        sigmoid_inputs = F.sigmoid(inputs)

        # This is here just to see the images as they go through the model -- remove later
        # torchvision.utils.save_image(sigmoid_inputs, r'C:\Users\Parinita Edke\Desktop\Presentation\inputs.png')
        # torchvision.utils.save_image(targets, r'C:\Users\Parinita Edke\Desktop\Presentation\targets.png')

        print(sigmoid_inputs.shape, targets.shape)

        # For calculating per image dice
        for i in range(sigmoid_inputs.shape[0]):
            pred = sigmoid_inputs[i, :, :, :]
            target = targets[i, :, :, :]
            print(pred.shape, pred.view(-1).shape, target.shape, target.reshape(-1).shape)

            pred = pred.view(-1)
            target = target.reshape(-1)

            img_intersection = (pred * target).sum()
            image_dice = (2. * img_intersection + smooth) / (pred.sum() + target.sum() + smooth)
            print(pred.sum(), target.sum())
            print((2. * img_intersection + smooth), (pred.sum() + target.sum() + smooth))

            image_dice_lst.append(image_dice.detach().cpu().numpy().item())

        # flatten label and prediction tensors
        sigmoid_inputs = sigmoid_inputs.view(-1)
        targets = targets.reshape(-1)

        intersection = (sigmoid_inputs * targets).sum()
        dice = (2. * intersection + smooth) / (sigmoid_inputs.sum() + targets.sum() + smooth)

        # print(f'Sum of Product of inputs & targets: {(inputs * targets).sum()}')
        # print(f'Numerator: {(2. * intersection + smooth)} , Denominator: {(inputs.sum() + targets.sum() + smooth)}')
        print(f'Dice score = {dice}')

        # Implement Sana's suggestion of penalizing segmentation outside of the fetus
        if baby_mask is not None:
            # 1. Get predictions of anomalies
            anomaly_preds = input_copy

            # 2. Set a tensor of zeros to 1 where (anomaly==1)&(baby==0)
            inverted_mtx = torch.zeros(anomaly_preds.shape, dtype=int).to(DEVICE)
            inverted_mtx[baby_mask.cpu() == 0] = 1

            # print(inverted_mtx.shape, anomaly_preds.shape)

            # 3. Multiply the tensor with the predictions tensor and then sum (or take the average?)
            outside_preds = inverted_mtx * anomaly_preds

            total_outside = torch.mean(outside_preds.float()).item()
            print(f"Total segmentation outside the baby: {total_outside}")

        if baby_mask is not None:
            return (1 - dice + (lmda*total_outside)), image_dice_lst
        else:
            return (1 - dice), image_dice_lst

        # 1 - inside_Dice + 2*outside_Dice


class TwoChannelDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TwoChannelDiceLoss, self).__init__()

    def forward(self, inputs, targets, baby_mask=None, smooth=1, lmda=0.9):
        image_dice_lst = []
        input_copy = F.sigmoid(inputs)
        target_copy = targets

        # comment out if your model contains a sigmoid or equivalent activation layer
        sigmoid_inputs = F.sigmoid(inputs)

        # This is here just to see the images as they go through the model -- remove later
        # torchvision.utils.save_image(sigmoid_inputs, r'C:\Users\Parinita Edke\Desktop\Presentation\inputs.png')
        # torchvision.utils.save_image(targets, r'C:\Users\Parinita Edke\Desktop\Presentation\targets.png')

        print(sigmoid_inputs.shape, targets.shape)

        # For calculating per image dice
        for i in range(sigmoid_inputs.shape[0]):
            pred = sigmoid_inputs[i, :, :, :]
            target = targets[i, :, :, :]
            pred_baby, pred_nt = pred.split(1, dim=0)
            target_baby, target_nt = target.split(1, dim=0)

            pred_baby = pred_baby.view(-1)
            pred_nt = pred_nt.view(-1)
            target_baby = target_baby.reshape(-1)
            target_nt = target_nt.reshape(-1)

            baby_intersection = (pred_baby * target_baby).sum()
            nt_intersection = (pred_nt * target_nt).sum()
            baby_dice = (2. * baby_intersection + smooth) / (pred_baby.sum() + target_baby.sum() + smooth)
            nt_dice = (2. * nt_intersection + smooth) / (pred_nt.sum() + target_nt.sum() + smooth)
            image_dice = (baby_dice+nt_dice)/2

            image_dice_lst.append(image_dice.detach().cpu().numpy().item())

        # flatten label and prediction tensors
        baby_sigmoid, nt_sigmoid = sigmoid_inputs.split(1, dim=1)
        baby_targets, nt_targets = targets.split(1, dim=1)
        baby_sigmoid = baby_sigmoid.reshape(-1)
        nt_sigmoid = nt_sigmoid.reshape(-1)
        baby_targets = baby_targets.reshape(-1)
        nt_targets = nt_targets.reshape(-1)

        baby_intersection = (baby_sigmoid * baby_targets).sum()
        nt_intersection = (nt_sigmoid * nt_targets).sum()
        baby_dice = (2. * baby_intersection + smooth) / (baby_sigmoid.sum() + baby_targets.sum() + smooth)
        nt_dice = (2. * nt_intersection + smooth) / (nt_sigmoid.sum() + nt_targets.sum() + smooth)
        dice = (baby_dice+nt_dice)/2

        # print(f'Sum of Product of inputs & targets: {(inputs * targets).sum()}')
        # print(f'Numerator: {(2. * intersection + smooth)} , Denominator: {(inputs.sum() + targets.sum() + smooth)}')
        print(f'Baby dice score = {baby_dice}, NT dice score = {nt_dice}, overall dice = {dice}')

        return (1 - dice), image_dice_lst


# TODO: Need to figure out how to separate inside and outside dice
class WeightedSoftDiceLoss(nn.Module):
    def __init__(self, v1=0.1, v2=0.9):
        super(WeightedSoftDiceLoss, self).__init__()

        # 0 <= v1 <= v2 <= 1; v2 = 1 - v1;
        self.v1 = v1
        self.v2 = v2

    def forward(self, inputs, targets, baby_mask=None, smooth=1, lmda=0.9):
        image_dice_lst = []

        # For calculating per image dice
        for i in range(inputs.shape[0]):
            pred = inputs[i, :, :, :]
            target = targets[i, :, :, :]
            print(pred.shape, pred.view(-1).shape, target.shape, target.reshape(-1).shape)

            img_W = (target * (self.v2 - self.v1)) + self.v1
            img_G_hat = img_W * (2 * pred - 1)
            img_G = img_W * (2 * target - 1)

            # pred = pred.view(-1)
            # target = target.reshape(-1)

            img_intersection = (img_G_hat * img_G).sum()
            image_dice = (2. * img_intersection + smooth) / ((img_G_hat**2).sum() + (img_G**2).sum() + smooth)
            print(img_G_hat.sum(), img_G.sum())
            print((2. * img_intersection + smooth), ((img_G_hat**2).sum() + (img_G**2).sum() + smooth))
            print(f"{i} Image_dice: {image_dice}")
            image_dice_lst.append(image_dice.detach().cpu().numpy().item())

        # Calculate the weight vector from the mask
        W = (targets * (self.v2 - self.v1)) + self.v1

        G_hat = W * (2*inputs - 1)
        G = W * (2*targets - 1)

        intersection = (G_hat * G).sum()
        wsdice = (2. * intersection + smooth) / ((G_hat**2).sum() + (G**2).sum() + smooth)
        print(f"WSDice: {wsdice} | Loss: {1-wsdice}")
        # print(image_dice_lst)

        return 1 - wsdice, image_dice_lst


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        image_bce_list = []
        for i in range(inputs.shape[0]):
            pred = inputs[i, :, :, :]
            target = targets[i, :, :, :]
            image_bce = F.binary_cross_entropy(pred, target, reduction='mean')

            image_bce_list.append(image_bce.detach().cpu().numpy().item())

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return BCE, image_bce_list


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU
import numpy as np
import os
import cv2

# Change this to gt directory
gt_file = './gt/Final_Labels_Final_2/'
# change this to predictions directory
pred_file = './pred_mra/'


EPS = 1e-12

def change_image_format(label):
    final_label = np.zeros((label.shape[0], label.shape[1]))
    img = label[:, :, 0] + label[:, :, 1] + label[:, :, 2]
    #print(img)
    pos_empty = np.where(img == 0)

    final_label[pos_empty[0],pos_empty[1]] = 0

    pos_green = np.where(label[:, :, 1] == 255)
    final_label[pos_green[0],pos_green[1]] = 1

    pos_blue = np.where(label[:, :, 0] == 255)
    final_label[pos_blue[0],pos_blue[1]] = 2

    pos_red = np.where(label[:, :, 2] == 255)
    final_label[pos_red[0],pos_red[1]] = 3

    return final_label

def get_iou_dice(gt, pr, n_classes):
    gt = change_image_format(gt)
    pr = change_image_format(pr)

    class_wise_iou = np.ones(n_classes)
    class_wise_dice = np.ones(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        # mask_sum = np.sum(gt == cl) + np.sum(pr == cl)
        iou = float(intersection)/(union + EPS)
        dice = ((2*float(intersection))/((union + intersection) + EPS))
        class_wise_iou[cl] = iou
        class_wise_dice[cl] = dice

    where_0 = np.where(class_wise_iou == 0)
    class_wise_iou[where_0] = 1
    where_0 = np.where(class_wise_dice == 0)
    class_wise_dice[where_0] = 1
    return class_wise_iou, class_wise_dice

def get_pixel_accuracy(gt, pr, n_classes):
    gt = change_image_format(gt)
    pr = change_image_format(pr)

    pixels_matching = 0
    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        pixels_matching += intersection

    return pixels_matching/gt.size

def precision(gt, pr, n_classes):
    gt = change_image_format(gt)
    pr = change_image_format(pr)

    class_wise_precision = np.zeros(n_classes)
    for cl in range(n_classes):
        tp = (np.sum((gt == cl)*(pr == cl)))
        fp = (np.sum(pr == cl) - tp)

        score = (tp + EPS) / (tp + fp + EPS)
        class_wise_precision[cl] = score

    return class_wise_precision

def recall(gt, pr, n_classes):
    gt = change_image_format(gt)
    pr = change_image_format(pr)

    class_wise_recall = np.zeros(n_classes)
    for cl in range(n_classes):
        tp = (np.sum((gt == cl)*(pr == cl)))
        fn = (np.sum(gt == cl) - tp)

        score = (tp + EPS) / (tp + fn + EPS)
        class_wise_recall[cl] = score

    return class_wise_recall


total_iou = 0
total_dice = 0
total_pixacc = 0
total_recall = 0
total_prec = 0

no_files = len(os.listdir(pred_file))
for f in os.listdir(pred_file):

    filename = f
    gt = cv2.imread(gt_file+filename)
    pr = cv2.imread(pred_file+filename)

    iou, dice = get_iou_dice(gt, pr, 4)
    total_iou += iou
    total_dice += dice
    total_pixacc += get_pixel_accuracy(gt, pr, 4)
    total_recall += recall(gt, pr, 4)
    total_prec += precision(gt, pr, 4)

print(np.average(total_iou/no_files), np.average(total_dice/no_files), total_pixacc/no_files, np.average(total_recall/no_files), np.average(total_prec/no_files))
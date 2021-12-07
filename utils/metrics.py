import numpy as np

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

    class_wise_iou = np.zeros(n_classes)
    class_wise_dice = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        # mask_sum = np.sum(gt == cl) + np.sum(pr == cl)
        iou = float(intersection)/(union + EPS)
        dice = ((2*float(intersection))/(union + intersection) + EPS)
        class_wise_iou[cl] = iou
        class_wise_dice[cl] = dice
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



# Tests
# gt = np.zeros((60,60), dtype=np.int8)
# gt[:,:20] = 0
# gt[:,20:40] = 1
# gt[:,40:60] = 2

# pr = np.zeros((60,60), dtype=np.int8)
# pr[:,:10] = 0
# pr[:,10:30] = 1
# pr[:,30:60] = 2

# print(get_iou_dice(gt, pr, 3))
# print(get_pixel_accuracy(gt, pr, 3)*100)
# print(precision(gt, pr, 3))
# print(recall(gt, pr, 3))

import cv2

gt_file = './gt/Final_Labels_Final_2/'
pred_file = './pred_vggnet/'

filename = '11223.png'
gt = cv2.imread(gt_file+filename)
pr = cv2.imread(pred_file+filename)

print(get_iou_dice(gt, pr, 4))
print(get_pixel_accuracy(gt, pr, 4)*100)
print(precision(gt, pr, 4))
print(recall(gt, pr, 4))
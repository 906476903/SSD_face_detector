import sys
import numpy as np
import cv2
import os
import numpy.random as npr
anno_file = "data/face/wider_face_train.txt"
im_dir = "/disk1/chengw/DataSet/WIDER/WIDER_train/images"
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
num_label = 0
img_dir = "data/VOCdevkit/face/JPEGImages/"
img_roi_dir = "data/VOCdevkit/face/JPEGImages_roi/"
label_dir = "data/VOCdevkit/face/Label/"
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    bbox = map(float, annotation[1:]) 
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    num_boxes = boxes.shape[0]
    save_img_dir = img_dir + str(num_label) + ".jpg"
    save_img_roi_dir = img_roi_dir + str(num_label) + ".jpg"
    save_label_dir = label_dir + str(num_label) + ".txt"

    img_shape = img.shape
    x_off = 0
    y_off = 0
    min_wh = min(img_shape[0], img_shape[1])
    if img_shape[0] > img_shape[1]:
        y_off = int((img_shape[0]-img_shape[1])/2)
    else:
        x_off = int((img_shape[1]-img_shape[0])/2)
    imgnew = img[y_off:y_off+min_wh-1, x_off:x_off+min_wh-1, :].copy()
    
    num_boxes1 = 0
    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        x1 = max(int(x1) - x_off, 0)
        x2 = min(int(x2) - x_off, imgnew.shape[1])
        y1 = max(int(y1) - y_off, 0)
        y2 = min(int(y2) - y_off, imgnew.shape[0])
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        if x1 < 0 or y1 < 0 or x2 > imgnew.shape[1]-1 or y2 > imgnew.shape[0]-1 or w < 10 or h < 10:
            continue
        num_boxes1 = num_boxes1 + 1
    if num_boxes1 < 1:
        continue
    cv2.imwrite(save_img_dir, imgnew)
    ff = open(save_label_dir, 'w')
    ff.write("%d\n"%num_boxes1)

    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        x1 = max(int(x1) - x_off, 0)
        x2 = min(int(x2) - x_off, imgnew.shape[1])
        y1 = max(int(y1) - y_off, 0)
        y2 = min(int(y2) - y_off, imgnew.shape[0])
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        if x1 < 0 or y1 < 0 or x2 > imgnew.shape[1]-1 or y2 > imgnew.shape[0]-1 or w < 10 or h < 10:
            continue
        ff.write("face %d %d %d %d\n"%(x1, y1, x2, y2))
        cv2.rectangle(imgnew,(x1,y1),(x2,y2),(255,255,0),2)
    ff.close()
    cv2.imwrite(save_img_roi_dir, imgnew)
    num_label = num_label+1

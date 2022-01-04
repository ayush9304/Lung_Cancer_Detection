import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import measure
from tqdm import tqdm
import random
import sys

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

model = tf.keras.models.load_model("UNet_Model.h5", custom_objects={'dice_coef':dice_coef, 'dice_coef_loss':dice_coef_loss})

def preprocess(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (512,512))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_improved = clahe.apply(img.astype(np.uint8))
    # img_improved = img.copy()
    centeral_area = img[100:400, 100:400]
    kmeans = KMeans(n_clusters=2).fit(np.reshape(centeral_area, [np.prod(centeral_area.shape), 1]))
    centroids = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centroids)
    ret, lung_roi = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    lung_roi = cv2.erode(lung_roi, kernel=np.ones([4,4]))
    lung_roi = cv2.dilate(lung_roi, kernel=np.ones([13,13]))
    lung_roi = cv2.erode(lung_roi, kernel=np.ones([8,8]))
    
    labels = measure.label(lung_roi)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0] < 475 and B[3]-B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)
    lung_roi_mask = np.zeros_like(labels)
    for N in good_labels:
        lung_roi_mask = lung_roi_mask + np.where(labels == N, 1, 0)
        
    contours, hirearchy = cv2.findContours(lung_roi_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    external_contours = np.zeros(lung_roi_mask.shape)
    for i in range(len(contours)):
        if hirearchy[0][i][3] == -1:  #External Contours
            area = cv2.contourArea(contours[i])
            if area>518.0:
                cv2.drawContours(external_contours,contours,i,(1,1,1),-1)
    external_contours = cv2.dilate(external_contours, kernel=np.ones([4,4]))
    
    external_contours = cv2.bitwise_not(external_contours.astype(np.uint8))
    external_contours = cv2.erode(external_contours, kernel=np.ones((7,7)))
    external_contours = cv2.bitwise_not(external_contours)
    external_contours = cv2.dilate(external_contours, kernel=np.ones((12,12)))
    external_contours = cv2.erode(external_contours, kernel=np.ones((12,12)))
    
    contours, hirearchy = cv2.findContours(external_contours,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    external_contours2 = np.zeros(external_contours.shape)
    for i in range(len(contours)):
        if hirearchy[0][i][3] == -1:  #External Contours
            area = cv2.contourArea(contours[i])
            if area>518.0:
                cv2.drawContours(external_contours2,contours,i,(1,1,1),-1)
    
    img_improved = img_improved.astype(np.uint8)
    external_contours2 = external_contours2.astype(np.uint8)
    extracted_lungs = cv2.bitwise_and(img_improved, img_improved, mask=external_contours2)
    
    return ((extracted_lungs-127.0)/127.0).astype(np.float32)

img = cv2.imread(sys.argv[1],0)
img = preprocess(img)
pred = model.predict(np.reshape(img,(1,512,512,1)))
pred = np.squeeze(pred)
pred[pred<0.5] = 0
pred[pred>=0.5] = 1

f = plt.figure(1)
plt.title("Original")
plt.imshow(img, cmap="gray")
f.show()
g = plt.figure(2)
plt.title("Mask")
plt.imshow(pred, cmap="gray")
g.show()
h = plt.figure(3)
plt.title("Highlighted")
ip = cv2.addWeighted(img,.4,pred,.6,0)
plt.imshow(ip, cmap="gray")
h.show()
input()
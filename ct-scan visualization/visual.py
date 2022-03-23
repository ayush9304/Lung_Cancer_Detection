import cv2
import numpy as np
import SimpleITK as stk
import matplotlib.pyplot as plt
from tqdm import tqdm

lungs_file = "1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd"
root = "DATA/"

def load_mhd(file):
    mhdimage = stk.ReadImage(file)
    ct_scan = stk.GetArrayFromImage(mhdimage)
    return ct_scan

# fourcc = cv2.VideoWriter_fourcc(*'MPEG')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

vid = cv2.VideoWriter('lungs.mp4', fourcc, 15.0, (512,512), False)

try:
    ct = load_mhd(root+lungs_file)
except:
    print("CT Scan file not found.\nExiting...")
    exit(0)

for i in tqdm(range(ct.shape[0])):
    img = ct[i,:,:]
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.resize(img, (512,512)).astype(np.uint8)
    cv2.imshow("lungs", img)
    # cv2.waitKey(1)
    vid.write(img)

vid.release()
cv2.destroyAllWindows()

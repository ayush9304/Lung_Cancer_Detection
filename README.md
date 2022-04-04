# Lung Cancer Detection

Lung cancer is the leading cause of cancer-related death worldwide. Screening high risk individuals for lung cancer with low-dose CT scans is now being implemented in the United States and other countries are expected to follow soon. In CT lung cancer screening, many millions of CT scans will have to be analyzed, which is an enormous burden for radiologists. Therefore there is a lot of interest to develop computer algorithms to optimize screening.
A vital first step in the analysis of lung cancer screening CT scans is the detection of pulmonary nodules, which may or may not represent early stage lung cancer.

This project is about segmentation of nodules in CT scans using 2D U-Net Convolutional Neural Network architecture.

## Dependencies
- numpy
- pandas
- opencv
- SimpleITK
- scikit-learn
- scikit-image
- tensorflow
- matplotlib

<!-- ## Run on custom image
```
py custom_image_input.py <image>
```
replace ```<image>``` with image location -->

## Dataset
The [LUNA16](https://luna16.grand-challenge.org/) dataset has been used in this project. 

## Preprocessing
The CT-Scans images passed through various preprocessing steps before inputting it to U-Net model for more accurate result. These processes includes segmenting the ROI (the lungs) from the surrounding regions of bones and fatty tissues. These include
- Binary Thresholding
- Erosion & Dilation for for removing noise
- Filling Holes by contours
- Extracting Lungs
- Extracting nodule masks

<p align="center">
<img width="737" alt="preprocessing_steps" src="https://user-images.githubusercontent.com/56977388/159730154-3681fc46-ca6c-4862-b779-1abb8b480887.png">
</p>

## Training
I used a 2D UNet convolutional neural network architecture which is mainly used for 
image segmentation. U-net is an encoder-decoder deep learning model which is known to 
be used in medical images. It is first used in biomedical image segmentation. U-net 
contained three main blocks, downsampling, upsampling, and concatenation. 
The dice coefficient loss is selected as the loss function. Dice coefficient as is often used 
in medical image segmentation.

<p align="center">
<img width="680" alt="unet architecture" src="https://user-images.githubusercontent.com/56977388/148122554-fdd46ffb-97ac-4cd3-807b-25a2c1b405fa.png">
</p>

This model was able to achieve a dice score of <code>0.81</code> in training data and <code>0.68</code> on test data. The model was trained for 94 epochs.

<p align="center">
<img width="600" alt="unet accuracy" src="https://user-images.githubusercontent.com/56977388/148122622-71cf02be-11f1-4997-9d8d-6ab0ee497ff2.png">
</p>

<!-- ![image](https://user-images.githubusercontent.com/56977388/148122622-71cf02be-11f1-4997-9d8d-6ab0ee497ff2.png) -->

## Result

<p align="center">
<img alt="result" src="https://user-images.githubusercontent.com/56977388/148122681-983d9e70-e5b6-4081-9fb7-233b5941bf9c.png">
<!-- ![image](https://user-images.githubusercontent.com/56977388/148122681-983d9e70-e5b6-4081-9fb7-233b5941bf9c.png) -->
</p>

## License

Licensed under the MIT License.

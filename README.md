# TRIPLE VIEW NET
Triple View - 2.5D method for pots inference segmentation tune up on cross-sectional medical image scans.

# Origin:

In medical cross-sectional sacns, images are typically stored as *.nii or *.nii.gz files which are 3D images. Therefore, 3D segmentation models are usually preferred for segmenting medical images due to their ability to retain information and build meaningful connections between slices. However, 3D segmentation Deep Learning networks often contain large number of parameters, making it extremely difficult to train and inference precisely and quickly without massive computational power, making them impractical for many medical facilities. 

# Solution: 

We propose a 2.5D network that aims to retain information between 2D slices of medical scans using 3 2D Deep Learning segmentation model based on 3 different medical image views: "coronal", "saggital" and "axial" to reduce the complexity of the network  while preserving the high precision of 3D network. When examining 2D slices taken from a medical scan image, each view has a different slicing direction which leads to different dice coefficient and accuracy for each models on each part of the slices. By using 3 models on 3 different views, we ensure that all parts of the images are segmented at all possible directions which helps in improving the segmented result. 
Our 2.5F network - named VAS - will work by first doing 2D segmentation on the 3 separate views of a medical image scan. Then, we use a voting system for each voxel from each models. If there is a tie in a voting system, we determine that voxel's class by comparing the dice score metric at that voxel for each model. For example, if at point(1, 2, 3) the axial model predicted it to be of class 1, but saggital model has a result of 2, we will compare dice_score[axial][1] > dice_score[saggital][2] ? axial : saggital.  


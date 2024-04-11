from patchify import patchify
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(
    "E:\\ISEF\\Triple-View-R-Net\\data_for_training\\images\\synapse0-slice006_axial.png", cv2.IMREAD_GRAYSCALE)

image = image/255.0
image = image.astype(np.float32)
# image = np.expand_dims(image, axis=0)
# patch_shape = (1, 16, 16)
patch_shape = (16, 16)
patches = patchify(image, patch_shape, 16)
patches = np.reshape(patches, (256, 256))
patches = patches.astype(np.float32)

# each element is a patch
# print(patches.shape)

# plt.imshow(patches)
# plt.show()

# Calculate the number of rows and columns in the patches grid
num_rows, num_cols = (16, 16)

# Create a new figure for plotting
plt.figure(figsize=(10, 10))

# Iterate over each patch and plot it
for i in range(num_rows):
    for j in range(num_cols):
        patch = patches[i][j]
        print(f"Patch shape: {patch.shape}")
        plt.subplot(num_rows, num_cols, i * num_cols + j + 1)
        plt.imshow(patch, cmap='gray')
        plt.axis('off')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

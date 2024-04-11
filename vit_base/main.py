import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import random

if __name__ == "__main__":

    # input image: H x W x C
    # output image: N x P x P
    # D = constant embedding size = p*p*c

    """Calculate input, output shape and hidden dims"""
    height = 256
    width = 256
    channel = 1
    p = 16
    N = (height * width) // (pow(p, 2))

    embedding_layer_input_shape = (height, width, channel)
    embedding_layer_output_shape = (N, p**2 * channel)

    print("Input shape", ":", embedding_layer_input_shape)
    print("Output shape", ":", embedding_layer_output_shape)

    """Patch images"""
    image = cv2.imread(
        "../data_for_training/images/synapse0-slice006_axial.png", cv2.IMREAD_GRAYSCALE)
    image = torch.from_numpy(image).unsqueeze(0)
    image_permuted = image.permute(1, 2, 0)

    # plt.figure(figsize=(p, p))
    # plt.imshow(image_permuted[:p, :p, :])

    """Set up code to plot top row as patches"""
    image_size = width
    patch_size = p
    num_patches = int(image_size / patch_size)

    # fig, axes = plt.subplots(nrows=1, ncols=num_patches,
    #                          sharex=True, sharey=True, figsize=(patch_size, patch_size))

    # # iterate through number of patches
    # for i, patch in enumerate(range(0, image_size, patch_size)):
    #     axes[i].imshow(image_permuted[:patch_size,
    #                    patch:patch+patch_size, :], cmap="gray")
    #     axes[i].set_xlabel(i + 1)
    #     axes[i].set_xticks([])
    #     axes[i].set_yticks([])

    """Set up code to plot whole image as patches"""
    # fig, axes = plt.subplots(
    #     nrows=num_patches, ncols=num_patches, figsize=(num_patches, num_patches), sharex=True, sharey=True)

    # for i, patch_height in enumerate(range(0, image_size, patch_size)):
    #     for j, patch_width in enumerate(range(0, image_size, patch_size)):
    #         axes[i][j].imshow(
    #             image_permuted[patch_height:patch_height + patch_size, patch_width:patch_width+patch_size, :])

    #         # set up label information
    #         axes[i, j].set_ylabel(
    #             i + 1, rotation="horizontal", horizontalalignment="right", verticalalignment="center")
    #         axes[i, j].set_xlabel(j + 1)
    #         axes[i, j].set_xticks([])
    #         axes[i, j].set_yticks([])
    #         axes[i, j].label_outer()

    # plt.show()

    """Turn patches into embedding using convolutional layer"""
    convlayer = nn.Conv2d(1, p*p*channel, kernel_size=p,
                          stride=p, padding=0)
    # pass image through convlayer
    image = image.unsqueeze(0).type(torch.float32)
    print(f"In: {image.shape}")
    image_out = convlayer(image)
    print(f"Out: {image_out.shape}")

    # visualize random feature maps of whole image that is convoluted into patches
    # random_indices = random.sample(range(0, 256), k=5)

    # fig, axes = plt.subplots(nrows=1, ncols=5)
    # for i in range(5):
    #     axes[i].imshow(image_out[0, random_indices[i], :, :].detach().numpy())
    #     axes[i].set_xlabel(i)
    #     axes[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    # plt.show()

    """Flatten image using nn.Flatten"""
    # output of convolutional embedding: 1, 256, 16, 16 => need to flatten out to 1, num_patches, embed_dims
    flattenlayer = nn.Flatten(start_dim=2, end_dim=3)
    image_out_flatten = flattenlayer(image_out)
    image_out_flatten_permuted = image_out_flatten.permute(0, 2, 1)
    print(f"Final output shape: {image_out_flatten_permuted.shape}")

    # visualize a single row from embedding image after flattening
    # fig, axes = plt.subplots(nrows=1, ncols=16, figsize=(
    #     16, 16), sharex=True, sharey=True)
    # for i, patch_height in enumerate(range(0, 256, 16)):
    #     axes[i].imshow(image_out_flatten_rearrange.permute(
    #         1, 2, 0).detach().numpy()[0, patch_height:patch_height+16, :], cmap="gray")
    #     axes[i].set_xlabel(i + 1)
    #     axes[i].set_xticks([])
    #     axes[i].set_yticks([])

    # plt.show()

    # visualize a single flattened feature map
    # single_flattend_map = image_out_flatten_permuted[:, :, 0]
    # plt.imshow(single_flattend_map.detach().numpy())
    # plt.title(f"Flattened feature map: {single_flattend_map.shape}")
    # plt.axis(False)
    # plt.show()

    """Turning patch embedding layer into pytorch module"""
    class PatchEmbedding(nn.Module):
        def __init__(self, channel=1, image_size=256, patch_size=16):
            super().__init__()
            embed_dim = patch_size * patch_size * channel
            self.conv = nn.Conv2d(
                channel, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)  # (batch_size, embed_dim, patch_size, patch_size)
            self.flatten = nn.Flatten(start_dim=2, end_dim=3)

        def forward(self, x):
            x = self.conv(x)
            x = self.flatten(x)
            x = x.permute(0, 2, 1)
            return x

    patcher = PatchEmbedding()
    print(f"Input image: {image.shape}")
    patched_image = patcher(image)
    print(f"Patch embedded image: {patched_image.shape}")

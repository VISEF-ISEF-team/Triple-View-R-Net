import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from original_attention_dataset import get_loaders
from original_unet_attention import Attention_Unet, SegGradModel, Attention_Unet_Seg_Grad_Model
from feature_extractor import FeatureExtractor


def main():
    train_loader, val_loader = get_loaders()
    image, _ = next(iter(train_loader))
    checkpoint = torch.load(
        "./files/original_attention_unet_checkpoint.pth.tar")
    model = Attention_Unet()
    model.load_state_dict(checkpoint)
    grad_model = Attention_Unet_Seg_Grad_Model(model=model)

    y_pred = grad_model(image)

    # class_output shape: [image in batch, class to view]
    class_output = y_pred[0, 0]

    class_score_sum = class_output.sum()
    class_score_sum.backward(retain_graph=True)

    gradients = grad_model.get_activations_gradient()
    print(f"Gradient: {gradients.shape}")

    # pooled gradient
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    print(f"Pooled Gradient: {pooled_gradients.shape}")

    activations = grad_model.get_activations(image).detach()
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    print(f"Activations: {activations.shape}")

    heatmap = torch.mean(activations, dim=1)[0]
    print(f"Heatmap shape: {heatmap.shape}")

    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(
        heatmap, cv2.COLORMAP_JET)

    image0 = image[0]
    image0 = image0.permute(1, 2, 0)
    image0 = image0.numpy()
    image0 = cv2.cvtColor(image0, cv2.COLOR_GRAY2BGR)

    print(f"Image: {image0.shape}")

    superimposed_img = (heatmap / 255.0) * 0.3 + image0

    plt.imshow(superimposed_img)

    plt.show()


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import numpy as np
from original_unet_attention import Attention_Unet, Attention_Unet_Refracted


class FeatureExtractor():
    def __init__(self, model, gradient_position):
        self.model = model
        self.gradient_position = gradient_position
        self.activations = []
        self.gradients = []

    def activations_hook(self, grad):
        self.gradients.appned(grad)

    def get_activations_gradient(self):
        return self.gradients

    def get_readily_activations(self, x):
        return self.activations

    def __call__(self, x):
        for name, module in self.model._modules.items():
            x = module(x)

            if name in self.gradient_position:
                print(f"Name: {name} in wanted modules || Module: {module}")
                x.set_hook(self.activations_hook)
                self.activations.append(x)
            else:
                print(
                    f"Name: {name} not in wanted modules || Module: {module}")
        return x


def main():
    pass


if __name__ == "__main__":
    main()

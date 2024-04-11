import numpy as np
from skimage.transform import resize
import torch


class OneHotResizing():
    def __init__(self, num_class, categories="auto", dtype=np.float32):
        self.num_class = num_class
        self.categories = categories
        self.dtype = dtype

    def one_hot_encode(self, mask):
        if isinstance(mask, np.ndarray):
            pass
        elif isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        else:
            mask = np.array(mask)

        if self.categories == "auto":
            unique_values = sorted(np.unique(mask))
        else:
            unique_values = self.categories

        output = [np.where(mask == x, 1, 0) for x in unique_values]
        output = np.stack(output, axis=0)
        return output

    def one_hot_resize(self, mask, new_shape):
        "New shape should also include dimension for class"

        encoded_mask = self.one_hot_encode(mask)

        try:
            encoded_mask_reshape = resize(
                encoded_mask, new_shape, preserve_range=True, anti_aliasing=True)
        except Exception as e:
            print(f"Resizing operation not succesful, function returning None")
            print(f"{e}")
            return None

        encoded_mask_reshape = np.argmax(encoded_mask_reshape, axis=0)
        encoded_mask_reshape = encoded_mask_reshape.astype(self.dtype)

        return encoded_mask_reshape


if __name__ == "__main__":
    unique_values = [0, 1, 2, 3, 4, 5, 6, 7]
    ors = OneHotResizing(8, dtype=np.uint8, categories=unique_values)

    for i in range(200):
        path = f"../UNETR_MMWHS/files/masks/heartmaskencode0-slice{str(i).zfill(3)}_axial.npy"
        mask = np.load(path)
        new_mask = ors.one_hot_resize(
            mask, new_shape=(8, 128, 128))

        print(mask.shape, new_mask.shape)

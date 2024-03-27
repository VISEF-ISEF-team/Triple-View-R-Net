import os
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from patchify import patchify
from unetr_2d import build_unetr_2d
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# UNETR configuration
cf = {}
cf["image_size"] = 256
cf["num_classes"] = 11
cf["num_channels"] = 3
cf["num_layers"] = 12
cf["hidden_dim"] = 128
cf["mlp_dim"] = 32
cf["num_heads"] = 6
cf["dropout_rate"] = 0.1
cf["patch_size"] = 16
cf["num_patches"] = (cf["image_size"]**2) // (cf["patch_size"]**2)
cf["flat_patches_shape"] = (
    cf["num_patches"],
    cf["patch_size"] * cf["patch_size"] * cf["num_channels"]
)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_dataset(path):
    train_x = sorted(glob(os.path.join(path, "lapa_train", "images", "*.jpg")))
    train_y = sorted(glob(os.path.join(path, "lapa_train", "labels", "*.png")))

    valid_x = sorted(glob(os.path.join(path, "lapa_val", "images", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path, "lapa_val", "labels", "*.png")))

    test_x = sorted(glob(os.path.join(path, "lapa_test", "images", "*.jpg")))
    test_y = sorted(glob(os.path.join(path, "lapa_test", "labels", "*.png")))

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    image = image / 255.0

    # Processing to patches
    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(image, patch_shape, cf["patch_size"])
    patches = np.reshape(patches, cf["flat_patches_shape"])
    patches = patches.astype(np.float32)

    return patches

def read_mask(path):
    path = path.decode()
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (cf["image_size"], cf["image_size"]))
    mask = mask.astype(np.int32)
    return mask

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        y = tf.one_hot(y, cf["num_classes"])
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape(cf["flat_patches_shape"])
    y.set_shape([cf["image_size"], cf["image_size"], cf["num_classes"]])
    return x, y
        
def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds


# Seeding
np.random.seed(42)
tf.random.set_seed(42)

# Directory for storing files
create_dir("files")

# Hyperparameters
batch_size = 16
lr = 0.1
num_epochs = 1
model_path = os.path.join("files", "model.keras")
csv_path = os.path.join("files", "log.csv")

# RGB Code and Classes
rgb_codes = [
    [0, 0, 0], [0, 153, 255], [102, 255, 153], [0, 204, 153],
    [255, 255, 102], [255, 255, 204], [255, 153, 0], [255, 102, 255],
    [102, 0, 51], [255, 204, 255], [255, 0, 102]
]

classes = [
    "background", "skin", "left eyebrow", "right eyebrow",
    "left eye", "right eye", "nose", "upper lip", "inner mouth",
    "lower lip", "hair"
]

dataset_path = "D:\Documents\GitHub\Triple-View-R-Net\data\LaPa"
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

print(f"Train: \t{len(train_x)} - {len(train_y)}")
print(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
print(f"Test: \t{len(test_x)} - {len(test_y)}")

train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

model = build_unetr_2d(cf)
print(model.summary())
model.compile(
    loss="categorical_crossentropy", 
    optimizer=SGD(lr)
)

callbacks = [
    ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
]

model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=valid_dataset,
    callbacks=callbacks
)
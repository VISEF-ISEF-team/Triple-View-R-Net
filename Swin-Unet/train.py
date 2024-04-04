from PIL import Image
from config import sample_config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from networks.vision_transformer import SwinUnet
 
def train():
    # sample input
    input_image = Image.new('L', (224,224), color='white')  
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    input_image = preprocess(input_image).unsqueeze(0)
    
    # get config
    config = sample_config()
    model = SwinUnet(config=config, num_classes=config.DATA.NUM_CLASSES)
    output = model(input_image)
    
train()
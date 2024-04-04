import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from networks.vit_seg_modeling import VisionTransformer, CONFIGS
import matplotlib.pyplot as plt

def train():
    config = CONFIGS['ViT-B_16']
    num_classes = config['n_classes']

    # Load pre-trained weights (if available)
    # model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))

    model = VisionTransformer(config, num_classes=num_classes)
    model.eval()

    # path_img = '../../data/ImageCHD/res_images_png_512/image_0001_0127.png'
    # input_tensor = torch.tensor(np.array(
    #     Image.open(path_img) 
    # ) / 255.0 )
    
    input_image = Image.new('L', (224, 224), color='white')

    # Preprocess the input image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]), 
    ])
    
    input_tensor = preprocess(input_image).unsqueeze(0)

    with torch.no_grad():
        print(input_tensor.size())
        output = model(input_tensor)
        print(f"\n{output.size()}")
        
train()
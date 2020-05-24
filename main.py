import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import random

def get_prediction(img):
    with torch.no_grad():

        LABEL_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()

        transform = transforms.Compose([ transforms.ToTensor() ])
        img = transform(img)

        prediction = model([img])[0]

        classes = []
        masks = []

        scores = list(prediction['scores'].detach().numpy()) # converting to numpy array then to a list
        classes_list = list(prediction['labels'].detach().numpy()) # indices of detected labels
        masks_list = list((prediction['masks']>0.5).squeeze().detach().numpy())

        scores = [scores.index(i) for i in scores if i > 0.5] # indices of matches that have higher than 0.5 confidence

        for i in scores:
            classes.append(LABEL_NAMES[classes_list[i]])
            masks.append(masks_list[i])

        return classes, masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Image Manipulator')
    parser.add_argument('--primary_image', type=str, help='path to image which is to be modified')
    parser.add_argument('--secondary_image', type=str, help='path to image which will modify the primary image')
    parser.add_argument('--output', type=str, default='output.png', help='path to where the output image will be saved')
    opt = parser.parse_args()

    primary_img = Image.open(opt.primary_image).convert('RGB')
    secondary_img = Image.open(opt.secondary_image).convert('RGB')

    classes, masks = get_prediction(primary_img)

    secondary_img = secondary_img.crop((0, 0, masks[0].shape[1], masks[0].shape[0]))

    print('==========================================================')

    choice = set(classes)
    choice.add('background')

    print("Classes found: ", choice)
    input_class = str(input("Enter a class: "))

    mask = np.zeros_like(masks[0], dtype=np.uint8)

    for i, c in enumerate(classes):
        if c == input_class or input_class == 'background':
            mask += (masks[i].astype(np.uint8)) * 255

    if input_class == 'background':
        mask = (np.where(mask == 255, 0, 255)).astype(np.uint8)

    mask = Image.fromarray(mask)

    output = primary_img.copy()
    output.paste(secondary_img, (0, 0), mask)

    output.save(opt.output)

    # plt.figure(figsize=(20, 20))
    # plt.imshow(output)
    # plt.show()


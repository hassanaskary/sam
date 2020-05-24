import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
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
        boxes = []

        scores = list(prediction['scores'].detach().numpy()) # converting to numpy array then to a list
        classes_list = list(prediction['labels'].detach().numpy()) # indices of detected labels
        masks_list = list((prediction['masks']>0.5).squeeze().detach().numpy())
        boxes_list = list(prediction['boxes'].detach().numpy())

        scores = [scores.index(i) for i in scores if i > 0.5] # indices of matches that have higher than 0.5 confidence

        for i in scores:
            classes.append(LABEL_NAMES[classes_list[i]])
            masks.append(masks_list[i])
            boxes.append([(boxes_list[i][0], boxes_list[i][1]), (boxes_list[i][2], boxes_list[i][3])])

        return classes, masks, boxes

def get_choice_img(primary_img, classes, boxes):
    choice_img = primary_img.copy()
    
    draw = ImageDraw.Draw(choice_img)
    for i in range(len(classes)):
        draw.rectangle((boxes[i][0], boxes[i][1]))
        draw.text(boxes[i][0], f"{classes[i]} {i}", fill='red', font=ImageFont.truetype("opensans.ttf", 20))
    
    return choice_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Image Manipulator')
    parser.add_argument('--primary_image', type=str, help='path to image which is to be modified')
    parser.add_argument('--secondary_image', type=str, help='path to image which will modify the primary image')
    parser.add_argument('--output', type=str, default='output.png', help='path to where the output image will be saved')
    opt = parser.parse_args()

    primary_img = Image.open(opt.primary_image).convert('RGB')
    secondary_img = Image.open(opt.secondary_image).convert('RGB')

    classes, masks, boxes = get_prediction(primary_img)

    if secondary_img.width < primary_img.width or secondary_img.height < primary_img.height:
        secondary_img = secondary_img.resize(
            (int(primary_img.width * 1.2), int(primary_img.height * 1.2))
        )

    secondary_img = secondary_img.crop((0, 0, masks[0].shape[1], masks[0].shape[0]))

    print('==========================================================')

    choice = set(classes)
    choice.add('background')

    choice_img = get_choice_img(primary_img, classes, boxes)
    plt.imshow(choice_img)
    plt.show()

    print("Classes found: ", choice)
    selection = str(input("Enter class names or instance numbers (comma seperated): "))
    selection = selection.split(',')
    selection = [x.strip() for x in selection]
    selection = [int(x) if x.isdigit() else x for x in selection]

    mask = np.zeros_like(masks[0], dtype=np.uint8)

    for i, c in enumerate(classes):
        if c in selection or i in selection:
            mask += (masks[i].astype(np.uint8)) * 255

    if 'background' in selection:
        background = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            background += (m.astype(np.uint8)) * 255
        background = (np.where(background == 255, 0, 255)).astype(np.uint8)
        mask += background

    mask = Image.fromarray(mask)

    output = primary_img.copy()
    output.paste(secondary_img, (0, 0), mask)

    output.save(opt.output)

    # plt.figure(figsize=(20, 20))
    # plt.imshow(output)
    # plt.show()


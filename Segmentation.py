
import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.transforms import get_affine_transform
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset
from collections import OrderedDict
dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}

class Person_body:
    def __init__(self,dataset="lip"): #dataset ['lip', 'atr', 'pascal']
        self.weight_path ="weights/checkpoint_60.pth.tar"
        # self.weight_path="weights/pascal.pth"
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = dataset_settings[dataset]['num_classes']
        self.input_size = dataset_settings[dataset]['input_size']
        self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]
        self.label = dataset_settings[dataset]['label']
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
                         ])
        self.palette = self.get_palette(self.num_classes)
        self.load_model()
        
    
    def load_model(self):
        self.model = networks.init_model('resnet18', num_classes=self.num_classes, pretrained=None)
        state_dict = torch.load(self.weight_path)['state_dict']
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
 

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def prepare(self, img):
        h, w, _ = img.shape

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.input_size)
        input = cv2.warpAffine(
            img,
            trans,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        input = self.transform(input)
        input=torch.unsqueeze(input, 0)
        meta = {
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return input, meta


    def get_palette(self,num_cls):
        """ Returns the color map for visualizing the segmentation mask.
        Args:
            num_cls: Number of classes
        Returns:
            The color map
        """
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette


    def detect(self,image):

        
        image, meta = self.prepare(image)
        c = meta['center']
        s = meta['scale']
        w = meta['width']
        h = meta['height']
        t1=time.time()
        
        output = self.model(image.to(self.device))
        print("time infer ",time.time()-t1)
        upsample = torch.nn.Upsample(size=self.input_size, mode='bilinear', align_corners=True)
        upsample_output = upsample(output[0][-1][0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
        
        logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=self.input_size)
      
        parsing_result = np.argmax(logits_result, axis=2)[:, ::-1]
        
        img_cv=np.asarray(parsing_result, dtype=np.uint8)
        # output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
        # output_img.putpalette(self.palette)
        # output_img.save("test.png")
        return img_cv

import time
if __name__ == '__main__':
    X=Segmentation()
    img=cv2.imread("inputs/000011.jpg",cv2.IMREAD_COLOR)
    for i in range(10):
        t1=time.time()
        X.detect(img)
        print("time ",time.time()-t1)

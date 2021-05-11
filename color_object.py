from webcolors import rgb_to_name
import time
from math import sqrt, atan2, degrees
import numpy as np
from scipy.spatial import KDTree
from color_var import Color_var
from webcolors import hex_to_rgb
import os
import glob
import cv2
from PIL import Image
from Get_color_object_with_mask.main import Color_object
class Color_Object:
    def __init__(self):
        self.color_object =Color_object()
        color_var=Color_var()
        self.hex_to_color= color_var.hex_to_color
        self.css3_names_to_hex = color_var.css3_names_to_hex
        
        self.css3_hex_to_names = self._reversedict(self.css3_names_to_hex)
        self.setup()

    def setup(self):
        css3_db = self.css3_hex_to_names
        self.names = []
        rgb_values = []
        for color_hex, color_name in css3_db.items():
            self.names.append(color_name)
            rgb_values.append(hex_to_rgb(color_hex))
        
        self.kdt_db = KDTree(rgb_values)
    def _reversedict(self,d):
        """
        Internal helper for generating reverse mappings; given a
        dictionary, returns a new dictionary with keys and values swapped.
        """
        return dict(zip(d.values(), d.keys()))
        

    def convert_rgb_to_names(self,rgb_tuple):
        
        # a dictionary of all the hex and their respective names in css3
    
        distance, index = self.kdt_db.query(rgb_tuple)

        return  self.hex_to_color[self.css3_names_to_hex[self.names[index]]]

    def predict_color_name(self,image,mask,list_mask,num_mask_idx=20):
        res=self.color_object.get_color(image=image,mask=mask,num_mask_idx=num_mask_idx,list_mask=list_mask,output="rgb")
        return self.convert_rgb_to_names(list(res[0]))

import time
if __name__=="__main__":
    x=Color_Object()
    label_mask=['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']

    datadir="images/train_images"
    labels="labels"
    image_paths=glob.glob(datadir+"/*")
    label_paths=glob.glob(labels+"/*")
    index=3
    image_path=image_paths[index]
    label_path=os.path.join(labels,image_path.split("/")[-1][:-3]+"png")
    label_img=cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
    print(image_path)
    image=cv2.imread(image_path)
    label=cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)[:, ::-1]
    t1=time.time()
    for i in range(1,len(label_mask)):
        try:
            print(label_mask[i]," : ",x.predict_color_name(image=image,mask=label,list_mask=[i]))
        except:
            print(label_mask[i]," : ","None")
    print("time ",time.time()-t1)
    cv2.imshow("image",image)
    cv2.waitKey(0)
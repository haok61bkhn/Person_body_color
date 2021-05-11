from Segmentation import  Person_body
from color_object import Color_Object
import cv2
class Person_body_color:
    def __init__(self):
        self.color_object = Color_Object()
        self.person_body = Person_body()
        self.label_body=['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']

    def predict(self,image):
        label=self.person_body.detect(image)
        cv2.imwrite("label.jpg",label)
        res={}
        for i in range(1,len(self.label_body)):
            try:
                # print(self.label_body[i]," : ", self.color_object.predict_color_name(image=image,mask=label,list_mask=[i]))
                res[self.label_body[i]]=self.color_object.predict_color_name(image=image,mask=label,list_mask=[i])
            except:
                # print(self.label_body[i]," : ","None")
                pass
        return res

if __name__=="__main__":
    x=Person_body_color()
  
    image = cv2.imread("b.jpg")
    print(x.predict(image))
    cv2.imshow("image",image)
    cv2.waitKey(0)
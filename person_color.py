from Segmentation import  Person_body
from color_object import Color_Object
import cv2
class Person_body_color:
    def __init__(self):
        self.color_object = Color_Object(2)
        self.person_body = Person_body()
        self.label_body=['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
        self.label2id={label:id for id,label in enumerate(self.label_body)}
        self.group_labels ={"upper_body":["Upper-clothes","Dress","Coat","Jumpsuits"],"lower_body":["Dress","Coat","Pants","Jumpsuits"]}

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

    def predicts(self,images):
        labels=self.person_body.detect_mutil(images)
        cv2.imwrite("label.jpg",labels)
        ress=[]
        for label in labels:
            res={}
            for i in range(1,len(self.label_body)):
                try:
                    # print(self.label_body[i]," : ", self.color_object.predict_color_name(image=image,mask=label,list_mask=[i]))
                    res[self.label_body[i]]=self.color_object.predict_color_name(image=image,mask=label,list_mask=[i])
                except:
                    # print(self.label_body[i]," : ","None")
                    pass
        return res

    def predict_group(self,image):
        label=self.person_body.detect(image)
        cv2.imwrite("label.jpg",label)
        res={}
        for gl in list(self.group_labels.keys()):
            try:
                ids=[]
                for lb in self.group_labels[gl]:
                    ids.append(self.label2id[lb])
                res[gl]=self.color_object.predict_color_name(image=image,mask=label,list_mask=ids)

            except Exception as e:
                print(e)


        return res
if __name__=="__main__":
    x=Person_body_color()
  
    image = cv2.imread("b.jpg")
    print(x.predict_group(image))
    cv2.imshow("image",image)
    cv2.waitKey(0)
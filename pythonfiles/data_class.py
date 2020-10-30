from torch.utils.data import Dataset
import numpy as np
import os
from torchvision import transforms
from PIL import Image


class SemanticDataset(Dataset):
    def __init__(self,train=False,val=False,test=False,transform=None):
        if(train):
            self.file_path_images=os.path.join((os.getcwd()),'Data/train/images/')
            self.file_path_labels=os.path.join((os.getcwd()),'Data/train/labels/')
            
        if(val):
            self.file_path_images=os.path.join((os.getcwd()),'Data/val/images/')
            self.file_path_labels=os.path.join((os .getcwd()),'Data/val/labels/')
            
        if(test):
            self.file_path_images=os.path.join((os.getcwd()),'Data/test/images/')
            self.file_path_labels=os.path.join((os.getcwd()),'Data/test/labels/')
            
        self.transform_image=transforms.Compose([transforms.Resize((240,240)),transforms.ToTensor()])
        self.transform_label=transforms.Compose([transforms.Resize((240,240)),transforms.ToTensor()])

        self.file_list_images=[os.path.splitext(filename)[0] for filename in os.listdir(self.file_path_images)]
        self.file_list_labels=[x+".png" for x in self.file_list_images]
        self.file_list_images=[x+".jpg" for x in self.file_list_images]
        
        print("Loaded ",len(self.file_list_images)," Images")


    def __len__(self):
        return len(self.file_list_images)

    def __getitem__(self,idx):
        
        image_path=os.path.join(self.file_path_images,self.file_list_images[idx])
        label_path=os.path.join(self.file_path_labels,self.file_list_labels[idx])
        
        image=Image.open(image_path)
        label=Image.open(label_path)
        #label=label.convert(mode='P')
        
        #r,g,b=label.split()
        
        return{'image':self.transform_image(image),'label':self.transform_label(label)}

    def show_image(self,idx):
        image_path=os.path.join(self.file_path_images,self.file_list_images[idx])
        label_path=os.path.join(self.file_path_labels,self.file_list_labels[idx])
        
        image=np.array(Image.open(image_path))
        image=Image.fromarray(image)
        image=image.convert('RGB')
        image.show()    


    def show_label(self,idx):
        label_path=os.path.join(self.file_path_labels,self.file_list_labels[idx])
        
        image=np.array(Image.open(label_path))*80.0
        image=Image.fromarray(image)
        image=image.convert('L')
        image.show()    
    

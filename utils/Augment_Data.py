import numpy as np
import cv2
import pandas as pd
import os
import glob
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import pickle
from google.colab.patches import cv2_imshow
from matplotlib import pyplot
import albumentations as A



def brightness_aug():
  num = 312
  for name in range(202,300):
    img_path  = "/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits/Dataset/Final_Images/{}.png".format(str(name))
    label_path = "/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits/Dataset/Final_Labels/{}.png".format(str(name))

    img = cv2.imread(img_path)
    label = cv2.imread(label_path)

        
    print(name)

    if(num%2 == 0):
      batch0 = tf.image.random_saturation(img, 5,10)
    else:
      batch0 = tf.image.random_brightness(img,0.4)

    batch0 = batch0.numpy()
        
    cv2.imwrite("/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits/Dataset/Augmented_Images/"+str(num) + ".png",batch0)
    cv2.imwrite("/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits/Dataset/Augmented_Labels/"+str(num) + ".png",label)
    num += 1
  


  







class MySequence(keras.utils.Sequence):
    def __init__(self,lst):
        self.lst = lst
        self.imgaug = ImageDataGenerator(rotation_range = 180,
                               horizontal_flip= True,
                               vertical_flip = True,
                               width_shift_range = 0.6,
                               height_shift_range = 0.6,
                               zoom_range = 0.4,
                               fill_mode = 'wrap'
                                )
    def __len__(self):
        return 10

    def get_augments(self):
      num = 113
      for name in range(1,200):
        img_path  = "/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits/Dataset/Final_Images/{}.png".format(str(name))
        label_path = "/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits/Dataset/Final_Labels/{}.png".format(str(name))

        img = cv2.imread(img_path)
        label = cv2.imread(label_path)

        #img = np.expand_dims(img,0)
        #label= np.expand_dims(label,0)
        print(name)

        
        params = self.imgaug.get_random_transform(img.shape)
        batch0 = self.imgaug.apply_transform(self.imgaug.standardize(img), params)
        batch1 = self.imgaug.apply_transform(self.imgaug.standardize(label), params)
        
        cv2.imwrite("/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits/Dataset/Augmented_Images/"+str(num) + ".png",batch0[:,:,:])
        cv2.imwrite("/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits/Dataset/Augmented_Labels/"+str(num) + ".png",batch1[:,:,:])
        num += 1



    def __getitem__(self, idx):
        X = np.array([cv2.resize(cv2.imread(self.path), (100, 100))
                      for _ in range(10)]).astype(np.float32)  # Fake batch of cats
        y = np.copy(X)
        for i in range(len(X)):
            # This creates a dictionary with the params
            params = self.imgaug.get_random_transform(X[i].shape)
            # We can now deterministicly augment all the images
            X[i] = self.imgaug.apply_transform(self.imgaug.standardize(X[i]), params)
            y[i] = self.imgaug.apply_transform(self.imgaug.standardize(y[i]), params)
        return X, y

def split_files_image(elem):
    elem = os.path.basename(elem)
    return int(elem.split(".")[0])

def split_files_label(elem):
    elem = os.path.basename(elem)
    return int(elem.split(".")[0])

def create_df(image_dir, label_dir):
    path = image_dir + '**/*'
    image_file_paths = glob.glob(path, recursive=True)
    path = label_dir + '**/*'
    label_file_paths = glob.glob(path, recursive=True)

    df = pd.DataFrame({'image': image_file_paths, 'label': label_file_paths})

    lst1 = df["image"].tolist()
    lst1.sort(key=split_files_image)

    lst2 = df["label"].tolist()
    lst2.sort(key=split_files_label)

    df = pd.DataFrame({'image': lst1, 'label': lst2}).astype(str)

    return df

def get_damaged_images(df:pd.DataFrame):
  lst = []
  for _,row in df.iterrows():
    img = cv2.imread(row['image'])
    label = cv2.imread(row['label'])
    val_red = np.where(np.all(label == (0, 0,255), axis=-1))
    val_blue = np.where(np.all(label == (255, 0,0), axis=-1))
    print("Executing")

    if (len(val_red[0]) > 0 or len(val_blue[0])>0):
      lst.append(os.path.basename(row['label']))
  

  return lst

def extract_random_patches(lst):
  
  for name in lst:
    img_path  = "/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits/Dataset/Final_Images/{}".format(name)
    label_path = "/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits/Dataset/Final_Labels/{}".format(name)
    img = cv2.imread(img_path)
    label = cv2.imread(label_path)

    img_lst = []
    label_lst = []
    
    val_red = np.where(np.all(label == (0, 0,255), axis=-1))
    val_blue = np.where(np.all(label == (255, 0,0), axis=-1))

    for i in range(0,len(val_red[0]),10):
      if(val_red[0][i] > 25 and val_red[0][i] < (512-26) and 
          val_red[1][i]> 25 and val_red[1][i]<(512-26)):
          img_lst.append(img[val_red[0][i]-25:val_red[0][i]+26,val_red[1][i]-25:val_red[1][i]+26,:])
          label_lst.append(label[val_red[0][i]-25:val_red[0][i]+26,val_red[1][i]-25:val_red[1][i]+26,:])
    
    for i in range(0,len(val_blue[0]),10):
      if(val_blue[0][i] > 25 and val_blue[0][i] < (512-26) and 
          val_blue[1][i]> 25 and val_blue[1][i]<(512-26)):
          img_lst.append(img[val_blue[0][i]-25:val_blue[0][i]+26,val_blue[1][i]-25:val_blue[1][i]+26,:])
          label_lst.append(label[val_blue[0][i]-25:val_blue[0][i]+26,val_blue[1][i]-25:val_blue[1][i]+26,:])
    

    return img_lst,label_lst


    
#Function to Rotate, Shear and flip images----->


'''df = create_df("/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits"
                    "/Dataset/Final_Images","/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits"
                    "/Dataset/Final_Images")
'''


#Reading files

with open ('/content/drive/MyDrive/Vision_Beyond_Limits_Dataset/VisionBeyondLimits/Dataset/damaged_buildings', 'rb') as fp:
    lst = pickle.load(fp)


print("Running")



brightness_aug()






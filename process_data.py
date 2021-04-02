
#Sort images to subfolders first 
import pandas as pd
import os
import shutil

#skin_df = pd.read_csv('data/HAM10000/HAM10000_metadata.csv')

data_dir = os.getcwd() + "/data/all_images/"
dest_dir = os.getcwd() + "/data/reorganized/"
skin_df2 = pd.read_csv('data/HAM10000/HAM10000_metadata.csv')


print(skin_df2['dx'].value_counts())

label=skin_df2['dx'].unique().tolist() 
label_images = []


# Copy images to new folders
for i in label:
    os.mkdir(dest_dir + str(i) + "/")
    sample = skin_df2[skin_df2['dx'] == i]['image_id']
    label_images.extend(sample)
    for id in label_images:
        shutil.copyfile((data_dir + "/"+ id +".jpg"), (dest_dir + i + "/"+id+".jpg"))
    label_images=[]    


        
from keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt


datagen = ImageDataGenerator()

train_dir = os.getcwd() + "/data/reorganized/"
train_data_keras = datagen.flow_from_directory(directory=train_dir,
                                         class_mode='categorical',
                                         batch_size=16,  #16 images at a time
                                         target_size=(32,32))  #Resize images


x, y = next(train_data_keras)

for i in range (0,15):
    image = x[i].astype(int)
    plt.imshow(image)
    plt.show()
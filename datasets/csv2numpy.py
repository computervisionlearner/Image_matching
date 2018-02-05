

import numpy as np
import cv2
import os
import glob
filename = 'water.csv'
dirname = filename.split('.')[0]
with open(filename,'r') as f:
  contents = f.readlines()
  
data = [row.split(',') for row in contents][1:]  #去掉表头

labels = []
images = []

all_pictures = sorted(glob.glob(os.path.join(dirname,'*.tiff')))
pictures_dicts = {}
for path in all_pictures:
  name = os.path.basename(path).split('.')[0]
  picture = cv2.imread(path,0)
  pictures_dicts[name] = picture
  
for row in data:
  type_ = row[0]
  rgb_name = row[1].split('.')[0] 
  rgb_x = int(row[2])
  rgb_y = int(row[3])
  nir_name = row[4].split('.')[0] 
  nir_x = int(row[5])
  nir_y = int(row[6])
  
  rgb_picture = pictures_dicts[rgb_name]
  nir_picture = pictures_dicts[nir_name]
  
  if type_ == 'positive':
    labels.append(1)
    assert (rgb_x,rgb_y) == (nir_x,nir_y)
  elif type_ == 'negative':
    labels.append(0)

  rgb_patch = rgb_picture[rgb_y-32:rgb_y+32,rgb_x-32:rgb_x+32]
  nir_patch = nir_picture[nir_y-32:nir_y+32,nir_x-32:nir_x+32]
  img = np.concatenate((rgb_patch,nir_patch),axis=1)  #[64,128]
  images.append(img)


images = np.expand_dims(np.array(images),axis=3)
images = np.concatenate(np.split(images,2,axis=2),axis=3)
labels = np.array(labels)


np.savez(dirname,images,labels)

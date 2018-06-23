import numpy as np
from scipy import misc as misc




images = np.squeeze(np.load("images.npy"))
labels = np.load("labels.npy")

left1, right1 = np.split(np.load("conv1_1.npy"),2,axis=2)

left2, right2 = np.split(np.load("conv2_1.npy"),2,axis=2)

left3, right3 = np.split(np.load("conv3_1.npy"),2,axis=2)

left4, right4 = np.split(np.load("conv3_2.npy"),2,axis=2)

left5, right5 = np.split(np.load("conv4_1.npy"),2,axis=2)

left6, right6 = np.split(np.load("conv4_2.npy"),2,axis=2)


features1 = np.load("features1.npy")
features2 = np.load("features2.npy")


  
diff1 = np.mean(left1-right1,axis=-1)
diff2 = np.mean(left2-right2,axis=-1)
diff3 = np.mean(left3-right3,axis=-1)
diff4 = np.mean(left4-right4,axis=-1)
diff5 = np.mean(left5-right5,axis=-1)
diff6 = np.mean(left6-right6,axis=-1)
diff7 = np.mean(features1-features2,axis=-1)
mean=np.mean(diff7,axis=(1,2))
 
i = 7
misc.imsave('diff1.jpg',diff1[i])
misc.imsave('diff2.jpg',diff2[i])
misc.imsave('diff3.jpg',diff3[i])
misc.imsave('diff4.jpg',diff4[i])
misc.imsave('diff5.jpg',diff5[i])
misc.imsave('diff6.jpg',diff6[i])  
misc.imsave('diff7.jpg',diff7[i])   
  

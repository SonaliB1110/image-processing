#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
image=cv2.imread('F1.png',0)
cv2.imshow('display image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


new_image=cv2.rotate(image,cv2.ROTATE_180)
cv2.imshow('display image',new_image)
cv2.waitKey(0)


# In[3]:


cv2.imwrite('D:\images\F1.png',image)
cv2.waitKey(0)


# In[4]:


print("image attributes",image.shape)


# In[5]:


#h,w,c=image.shape
#print("width:",w)
#print("height:",h)
#print("channel:",c)


# In[6]:


import matplotlib.pyplot as plt
images=plt.imread('F1.png')
plt.imshow(images)
plt.show()


# In[7]:


image.size


# In[8]:


print("image attributes",images.shape)


# In[9]:


ret, bw_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
  
# converting to its binary form
bw = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  
cv2.imshow("Binary", bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[10]:


down_width = 300
down_height = 200
down_points = (down_width, down_height)
resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
cv2.imshow('Resized Down by defining height and width', resized_down)
cv2.waitKey()
cv2.destroyAllWindows()


# In[11]:


up_width = 600
up_height = 400
up_points = (up_width, up_height)
# resize the image
resized_up = cv2.resize(image, up_points, interpolation = cv2.INTER_LINEAR)
cv2.imshow('Resized Up image by defining height and width', resized_up)
cv2.waitKey()
cv2.destroyAllWindows()


# In[12]:


print('The Shape of the image is:',image.shape)
print('The image as array is:')
print(image)


# In[13]:


from PIL import Image
import numpy as np
w, h = 512, 512
data = np. zeros((h, w, 3),dtype=np.uint8)
data[120:256, 120:256] = [255, 0, 0] # red patch in upper left.
img = Image.fromarray(data,'RGB')
img.save('my.png')


# In[14]:


images=plt.imread('my.png')
plt.imshow(images)
plt.show()


# In[15]:


import cv2
image=cv2.imread("F1.png")
color=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
cv2.imshow("display BGR",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[16]:


import cv2
image=cv2.imread("F1.png")
color=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
cv2.imshow("display BGR",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[17]:


import cv2
image=cv2.imread("F1.png")
color=cv2.cvtColor(image,cv2.COLOR_HLS2RGB)
cv2.imshow("display BGR",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[18]:


import cv2
image=cv2.imread("F1.png")
color=cv2.cvtColor(image,cv2.COLOR_LAB2RGB)
cv2.imshow("display BGR",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[19]:


import cv2
image=cv2.imread("F1.png")
color=cv2.cvtColor(image,cv2.COLOR_LAB2BGR)
cv2.imshow("display ",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[20]:


import cv2
image=cv2.imread("F1.png")
color=cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
cv2.imshow("display BGR",color)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





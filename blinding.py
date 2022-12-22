#!/usr/bin/env python
# coding: utf-8

# In[1]:


#converting 
from PIL import Image
img = Image.open('flower.jpg')
img.save("D:/flower.tiff",'TIFF')


# In[2]:


import numpy as np
from skimage import data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
image =data.camera()  
type(image)
np.ndarray

mask = image < 87  
image[mask]=255  
plt.imshow(image, cmap='gray') 


# In[3]:


import numpy as np
from skimage import data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
image =data.coins()  
type(image)
np.ndarray

mask = image < 87  
image[mask]=255  
plt.imshow(image, cmap='gray') 


# In[4]:


import numpy as np
from skimage import data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
image =data.clock()  
type(image)
np.ndarray

mask = image < 87  
image[mask]=255  
plt.imshow(image, cmap='gray') 


# In[5]:


from PIL import Image

# Function to change the image size
def changeImageSize(maxWidth,
                    maxHeight,
                    image):
   
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    newImage    = image.resize((newWidth, newHeight))
    return newImage
   
# Take two images for blending them together  
image1 = Image.open("F2.jpg")
image2 = Image.open("img3.jpg")

# Make the images of uniform size
image3 = changeImageSize(800, 500, image1)
image4 = changeImageSize(800, 500, image2)

# Make sure images got an alpha channel
image5 = image3.convert("RGBA")
image6 = image4.convert("RGBA")

# Display the images
image5.show()
image6.show()

# alpha-blend the images with varying values of alpha
alphaBlended1 = Image.blend(image5, image6, alpha=.2)
alphaBlended2 = Image.blend(image5, image6, alpha=.4)

# Display the alpha-blended images
alphaBlended1.show()
alphaBlended2.show()


# In[6]:


import cv2
image = cv2.imread("D:\images\krishna.jpg")

y=0
x=0
h=300
w=510
crop_image = image[x:w, y:h]
cv2.imshow("Cropped", crop_image)
cv2.waitKey(0)


# In[7]:


from PIL import Image

#Create an Image Object from an Image
im = Image.open('D:/images/flower.jpg')

#Display actual image
im.show()

#left, upper, right, lowe
#Crop
cropped = im.crop((1,2,500,500))

#Display the cropped portion
cropped.show()

#Save the cropped image
cropped.save('D:/images/croppedBeach1.jpg')


# In[8]:


import cv2
import numpy as np
# Load the image
img = cv2.imread('img3.jpg')
# Check the datatype of the image
print(img.dtype)
# Subtract the img from max value(calculated from dtype)
img_neg = 255 - img
# Show the image
cv2.imshow('negative',img_neg)
cv2.waitKey(0)


# In[9]:


from PIL import Image, ImageOps

im = Image.open('D:/images/img3.jpg')
im_invert = ImageOps.invert(im)
im_invert.save('D:/images/negation.jpg', quality=95)
im_invert.show()


# In[10]:


# Python3 program to draw line
# shape on solid image
import numpy as np
import cv2

# Creating a black image with 3 channels
# RGB and unsigned int datatype
img = cv2.imread('img3.jpg')

# Creating line
cv2.line(img, (20, 160), (600, 160), (0, 0, 255), 30)

cv2.imshow('dark', img)

# Allows us to see image
# until closed forcefully
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


# Python3 program to draw rectangle
# shape on solid image
import numpy as np
import cv2

# Creating a black image with 3
# channels RGB and unsigned int datatype
img = cv2.imread('img3.jpg')

# Creating rectangle
cv2.rectangle(img, (30, 30), (300, 200), (0, 255, 0), 5)

cv2.imshow('dark', img)

# Allows us to see image
# until closed forcefully
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[12]:


# Python3 program to draw circle
# shape on solid image
import numpy as np
import cv2

# Creating a black image with 3
# channels RGB and unsigned int datatype
img = cv2.imread('img3.jpg')

# Creating circle
cv2.circle(img, (200, 200), 80, (255, 0, 0), 3)

cv2.imshow('dark', img)

# Allows us to see image
# until closed forcefully
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


# Python3 program to write
# text on solid image
import numpy as np
import cv2

# Creating a black image with 3
# channels RGB and unsigned int datatype
img = cv2.imread('img3.jpg')

# writing text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'GOOD MORNING', (60, 60),
            font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('dark', img)

# Allows us to see image
# until closed forcefully
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[14]:


# importing required libraries of opencv
import cv2

# importing library for plotting
from matplotlib import pyplot as plt

# reads an input image
img = cv2.imread('img1.jpg',0)

# find frequency of pixels in range 0-255
histr = cv2.calcHist([img],[0],None,[256],[0,256])

# show the plotting graph of an image
plt.plot(histr)
plt.show()


# In[15]:


import cv2
from matplotlib import pyplot as plt
img = cv2.imread('img1.jpg',0)
 
# alternative way to find histogram of an image
plt.hist(img.ravel(),256,[0,256])
plt.show()


# In[16]:


import numpy as np
from skimage import data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
image =data.astronaut() 
plt.imshow(image)
plt.show()
type(image)
np.ndarray

mask = image < 87  
image[mask]=255  
plt.imshow(image,cmap='gray') 


# In[17]:


import numpy as np
from skimage import data
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
image =data.horse()  
plt.imshow(image)
plt.show()
type(image)
np.ndarray

mask = image < 87  
image[mask]=125  
plt.imshow(image,cmap='gray') 


# In[18]:


import numpy as np
from skimage import data
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
image =data.horse()  
viewer = ImageViewer(image)
viewer.show()


# In[19]:


import cv2

# read two input images.
# The size of both images must be the same.
img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')

# compute bitwise AND on both images
and_img = cv2.bitwise_and(img1,img2)

# display the computed bitwise AND image
cv2.imshow('Bitwise AND Image', and_img)
or_img = cv2.bitwise_or(img1,img2)

# display the computed bitwise OR image
cv2.imshow('Bitwise OR Image', or_img)
xor_img = cv2.bitwise_xor(img1,img2)

# display the computed bitwise XOR image
cv2.imshow('Bitwise XOR Image', xor_img)
bitwise_not = cv2.bitwise_not(img1,img2)

# display the computed bitwise NOT image
cv2.imshow("bitwise_not", bitwise_not)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[20]:


from PIL import Image,ImageStat
im=Image.open('img1.jpg')
stat=ImageStat.Stat(im)
print(stat.mean)


# In[21]:


from PIL import Image,ImageStat
im=Image.open('img1.jpg')
stat=ImageStat.Stat(im)
print(stat.stddev)


# In[22]:


from PIL import Image,ImageStat
im=Image.open('img1.jpg')
stat=ImageStat.Stat(im)
print(stat.median)


# In[23]:


#RGB Channels
import matplotlib.pyplot as plt
im1=Image.open("img3.jpg")
ch_r,ch_g,ch_b=im1.split()
plt.figure(figsize=(18,6))
plt.subplot(1,3,1);
plt.imshow(ch_r,cmap=plt.cm.Reds);plt.axis('off')
plt.subplot(1,3,2);
plt.imshow(ch_g,cmap=plt.cm.Greens);plt.axis('off')
plt.subplot(1,3,3);
plt.imshow(ch_b,cmap=plt.cm.Blues);plt.axis('off')
plt.tight_layout()
plt.show()


# In[26]:


import cv2
import numpy as np

# load image and get dimensions
img = cv2.imread("ut.jpg")
h, w, c = img.shape

# create zeros mask 2 pixels larger in each dimension
mask = np.zeros([h + 2, w + 2], np.uint8)

# do floodfill
result = img.copy()
cv2.floodFill(result, mask, (0,0), (255,255,255), (3,151,65), (3,151,65), flags=8)
cv2.floodFill(result, mask, (38,313), (255,255,255), (3,151,65), (3,151,65), flags=8)
cv2.floodFill(result, mask, (363,345), (255,255,255), (3,151,65), (3,151,65), flags=8)
cv2.floodFill(result, mask, (619,342), (255,255,255), (3,151,65), (3,151,65), flags=8)

# write result to disk
cv2.imwrite("me.png", result)

# display it
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[27]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("ut.jpg")

res = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)

plt.figure(figsize=(15,12))

plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(res,cmap = 'gray')
plt.title('Downsampled Image')

plt.show()


# In[28]:


#original image
import cv2
import matplotlib.pyplot as plt
from PIL import Image
im = Image.open("ut.jpg")
plt.imshow(im)
plt.show()

#up sampling
import cv2
import matplotlib.pyplot as plt
from PIL import Image
im = Image.open("ut.jpg")
im = im.resize((im.width*5, im.height*5), Image.NEAREST)
plt.figure(figsize=(10,10))
plt.imshow(im)
plt.show()

#down sampling
im = Image.open("ut.jpg")
im = im.resize((im.width//5, im.height//5))
plt.figure(figsize=(15,10))
plt.imshow(im)
plt.show()


# In[29]:


import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
im = Image.open("ut.jpg")
plt.figure(figsize=(20,30))
num_colors_list = [1 << n for n in range(8,0,-1)]
snr_list = []
i = 1
for num_colors in num_colors_list:
    im1 = im.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
    plt.subplot(4,2,i), plt.imshow(im1), plt.axis('off')
    snr_list.append(signaltonoise(im1, axis=None))
    plt.title('Image with # colors = ' + str(num_colors) + ' SNR = ' +
    str(np.round(snr_list[i-1],3)), size=20)
    i += 1
plt.subplots_adjust(wspace=0.2, hspace=0)
plt.show()


# In[30]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('s.jpg')
near_img = cv2.resize(img,None, fx = 5, fy = 5, interpolation = cv2.INTER_NEAREST)
cv2.imshow(" ",near_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np
img = cv2.imread('s.jpg')
near_img = cv2.resize(img,None, fx = 5, fy = 5, interpolation = cv2.INTER_LINEAR)
cv2.imshow(" ",near_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np
img = cv2.imread('s.jpg')
near_img = cv2.resize(img,None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
cv2.imshow(" ",near_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


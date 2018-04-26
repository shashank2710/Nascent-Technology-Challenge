# import the necessary packages
import cv2
import os, os.path
import numpy as np
#debug info OpenCV version
print ("OpenCV version: " + cv2.__version__)
 
#image path and valid extensions
imageDir = "training_data/Mountain_Bike" #specify your path here
image_path_list = []
valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] #specify your vald extensions here
valid_image_extensions = [item.lower() for item in valid_image_extensions]
 
#create a list all files in directory and
#append files with a vaild extention to image_path_list
for file in os.listdir(imageDir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))
 
#loop through image_path_list to open each image
for imagePath in image_path_list:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (512,512),0,0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    images = np.multiply(image, 1.0 / 255.0)
    
    cv2.imshow(imagePath, images)
    
    
    key = cv2.waitKey(0)
    if key == 27: # escape
        break
 
# close any open windows
cv2.destroyAllWindows()
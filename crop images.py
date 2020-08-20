"""
Function: Crops the images into small 256x256 images and divides the dataset into training and testing set.
 
"""
 
import numpy as np
import cv2
from tqdm import tqdm
import os
import math
import time
from google.colab.patches import cv2_imshow
 
def crop_and_save(images_path, masks_path, new_images_path, new_masks_path, img_width, img_height):
    """
    Imports Images and creates multiple crops and then stores them in the specified folder. Cropping is important in the project to protect spatial information, which otherwise would be lost if we resize the images.
    Please note:
    > All the images which has less than 1% annotation, in terms of area is removed. In other words, Images that are 99% empty are removed.
 
    Parameters
    ----------
    >images_path (str): Path to the directory containing all the images.
    >masks_path (str): Path to the directory containing all the masks.
    >new_images_path (str): Path to the Directory where the cropped images will be stored.
    >new_masks_path (str): Path to the Directory where the cropped masks will be stored.
    >img_width (int): width of the cropped image.
    >img_height (int): height of the cropped image.
    """
    print("Building Dataset.")
    num_skipped = 0
    start_time = time.time()
    files = next(os.walk(images_path))[2]
    # print(files)
    
    for image_file in tqdm(files, total = len(files)):
        image_path = images_path + image_file
        image = cv2.imread(image_path)
        
        
        mask_path = masks_path + image_file
        mask = cv2.imread(mask_path) 
        try:
            num_splits = math.floor((image.shape[0]*image.shape[1])/(img_width*img_height))
            counter = 0
            for r in range(0, image.shape[0], img_height):
                for c in range(0, image.shape[1], img_width):
                    counter += 1
                    blank_image = np.zeros((img_height ,img_width, 3), dtype = "uint8")
                    blank_mask = np.zeros((img_height ,img_width,3), dtype = "float32")
                    
                    new_image_path = new_images_path + str(counter) + '_' + image_file
                    new_mask_path = new_masks_path + str(counter) + '_' + image_file
                    new_image = np.asarray(image[r:r+img_height, c:c+img_width])                    
                    new_mask = np.asarray(mask[r:r+img_height, c:c+img_width])  
 
 
                    if(new_image.shape!=(256,256,3) or new_mask.shape!=(256,256,3)):
                        continue         
            
                    blank_image[:new_image.shape[0], :new_image.shape[1], :] += new_image
                    blank_mask[:new_image.shape[0], :new_image.shape[1]] += new_mask
                    
                    blank_mask[blank_mask>1] = 255
                    # Skip any Image that is more than 99% empty.
                    if np.any(blank_mask):
                        num_black_pixels, num_white_pixels = np.unique(blank_mask, return_counts=True)[1]
                        
                        if num_white_pixels/num_black_pixels < 0.01:
                            num_skipped+=1
                            continue
 
                        cv2.imwrite(new_image_path, blank_image)
                        cv2.imwrite(new_mask_path, blank_mask)
        except:
            continue
    print("EXPORT COMPLETE: {} seconds.\nImages exported to {}\nMasks exported to{}".format(round((time.time()-start_time), 2), new_images_path, new_masks_path))
    print("\n{} Images were skipped.".format(num_skipped))
 
if __name__ == "__main__":
    root_data_path = os.getcwd() + '/Data/'
    test_to_train_ratio = 0.3 
    img_width = img_height = 256
    num_channels = 3
 
    # Path Information
    images_path = root_data_path + "Images/"
    masks_path = root_data_path + "Targets/"
    new_images_path = root_data_path + "newImages/"
    new_masks_path = root_data_path + "newMasks/"
 
    for path in [new_images_path, new_masks_path]:
        if not os.path.exists(path):
            os.mkdir(path)
            print("DIRECTORY CREATED: {}".format(path))
        else:
             print("DIRECTORY ALREADY EXISTS: {}".format(path)) 
             
    crop_and_save(images_path, masks_path, new_images_path, new_masks_path, img_width, img_height)
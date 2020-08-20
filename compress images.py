from tqdm import tqdm_notebook
import os
import cv2
import h5py
import numpy as np


all_images = []
all_masks = []
def compress_images():
    """
    Imports images and respective masks and exports all of them into a h5py file.
    """
    root_path = os.getcwd()
    data_path = root_path +'/Data/'
    global all_images, all_masks
    rej_count = 0
    counter = 0
    files = next(os.walk('/content/drive/My Drive/ColabNotebooks/Data/newImages'))[2]

    print('Total number of files =',len(files))
    
    for image_file in tqdm_notebook(files, total = len(files)):
        counter += 1

        image_path = data_path+ 'croppedImages/'+image_file
        mask_path = data_path+ 'croppedMasks/'+image_file

        if not os.path.exists(image_path): continue  
        image = cv2.imread(image_path)   
        mask = cv2.imread(mask_path)  
        try:
            t1= image.shape
            t2= mask.shape
        except:
            print(image_path)
            continue
        all_images.append(image)  
        all_masks.append(mask)
    all_images = np.asarray(all_images)
    all_masks = np.asarray(all_masks)

    
    print('{} images were rejected.'.format(rej_count))
    print("Shape of Train Images =", all_images.shape)
    print("Shape of Train Labels =", all_masks.shape)
    print("Memory size of Image array = ", all_images.nbytes)
    print("Memory size of Image array = ", all_masks.nbytes)

    print("Data has been successfully exported.")
    with h5py.File(data_path+ 'road_masks.h5py', 'w') as hf:
        hf.create_dataset("all_masks",  data=all_masks) 
    with h5py.File(data_path+ 'road_images.h5py', 'w') as hf:
        hf.create_dataset("all_images",  data=all_images)   
                  
compress_images()
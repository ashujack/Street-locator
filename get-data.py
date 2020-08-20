##### for mounting drive on colab

# from google.colab import drive
# drive.mount('/content/drive')

#################################

import urllib.request
import os
import time
import tensorflow as tf
root_path = os.getcwd()

def download_images(link_file_images,output_directory,image_type):
  print("\nDownoading", image_type)

  with open(link_file_images,'r') as link_file:
    image_links = link_file.readlines()

  for idx, image_link in enumerate(image_links):
    image_path = output_directory + image_type  + "/image_%d.tiff" % (idx+1)  
    urllib.request.urlretrieve(image_link, image_path)


if __name__ == '__main__':    
    dataset_name = "MassachusettsRoads"
    link_file_images = (root_path+ "/src/Images.txt").format(dataset_name)
    link_file_targets = (root_path+ "/src/Targets.txt").format(dataset_name)
    output_directory = (root_path+ "/Data/").format(dataset_name)
    tf.io.gfile.mkdir(output_directory+ "Images")
    tf.io.gfile.mkdir(output_directory+ "Targets" )
    if not os.path.exists(output_directory):
        tf.io.gfile.mkdir(output_directory)
        

    start_time = time.time()
    download_images(link_file_images, output_directory, "Images")
    download_images(link_file_targets, output_directory, "Targets")
    print("TOTAL TIME: {} minutes".format(round((time.time() - start_time)/60, 2)))
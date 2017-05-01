#!/usr/bin/env python3
import cv2
import glob
import numpy as np
import os
import theano
import PIL.Image as Image
from scipy import ndimage
EXTENSIONS = ['png', 'jpg']
theano.config.floatX = 'float32'

def load_data(directories, mask = None):
   
    def load_images(directory):
        image_list = []
        for filename in glob.glob(directory):
            ext = filename.split('.')
            if ext[-1].lower() not in EXTENSIONS:
                continue
           # print(filename)
            image = cv2.imread(filename)
            image_list.append(image)
           
        return image_list
   
    def shared_dataset(dataset, mask, borrow = True):
        """
        Keep dataset in shared variables. This trick allows Theano
        to copy data into the GPU memory, when the code is run on the GPU.
        Since copying everything to the GPU is an overwhelming task, we'll
        copy mini-batches.
        """     

        if mask == None:
            scale = 0.25
            image = dataset[0]
            mask = np.ones(image.shape)
            l = int(scale * image.shape[0]) - 1
            u = int((1.0 - scale) * image.shape[0])
            mask[l:u, l:u, :] = 0.0

        # create input and output of the nn
        dataset_x=dataset
        dataset_y = dataset * mask
        dataset_z = dataset * (1 - mask)
        
#        noise_factor = 0.5
#        dataset_z_noisy=dataset_y+ noise_factor * np.random.normal(
#                loc=0.0, 
#                scale=0.3, 
#                size=dataset_z.shape
#                )
#        dataset_z_noisy=dataset_z_noisy * (1-mask)+dataset_y
       
        # make shared variables of input and output
        shared_x = np.asanyarray(dataset_x,
                                              dtype = theano.config.floatX)
        shared_y =np.asanyarray(dataset_y,
                                              dtype = theano.config.floatX)
        shared_z =np.asanyarray(dataset_z,
                                              dtype = theano.config.floatX)
        return shared_x, shared_y, shared_z
   
    image_list = []
    for directory in directories:
        images = load_images(directory)
        if len(image_list) == 0:
            image_list = np.array(images)
        else:
            image_list = np.concatenate((image_list, images), axis = 0)

    dataset_x, dataset_y, dataset_z = shared_dataset(image_list, mask)
    return dataset_x, dataset_y, dataset_z
       
def search_dir(directory):
    directories = [os.path.join(directory, '*')]
    for folder_name in os.listdir(directory):
        dir = os.path.join(directory, folder_name)
        if os.path.isfile(dir):
            continue
        dir = os.path.join(dir, '*')
        directories.append(dir)
    return directories
       
       
if __name__ == '__main__':
    path = './dataset/training'
    directories = search_dir(path)
    print(directories)
    dataset_x, dataset_y,dataset_z = load_data(directories)
    print dataset_x.shape.eval()
   
     
  

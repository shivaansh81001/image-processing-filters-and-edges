import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, img_as_ubyte,img_as_float, exposure,filters,feature
from scipy import signal, spatial

def noise_removal():
    '''This function loads the original noisy image and compares convolution of 2 filters - median and gaussian'''
    def print_images(original, median, gaussian):
        plt.figure(figsize=(10,5))
        plt.subplot(1, 3, 1), plt.imshow(original, cmap='gray'), plt.title("original")
        plt.subplot(1, 3, 2), plt.imshow(median , cmap='gray'), plt.title("medain ")
        plt.subplot(1, 3, 3), plt.imshow(gaussian , cmap='gray'), plt.title("gaussian")
        plt.tight_layout()
        plt.show()
        
    
    def load_image(filename):
        
        image = io.imread(filename ,as_gray= True)
        img = img_as_ubyte(image)
        return img
        
    
    def median_filter(img):
        return filters.median(img)
    
    def gaussian_filter(img):
        return filters.gaussian(img)
    
    img= load_image("assets/noisy.jpg")
    #io.imshow(img)
    
    median = median_filter(img)
    gaussian = gaussian_filter(img)
    
    print_images(img, median, gaussian)
    
    print(" As we know the salt and pepper noise in an image are nothing else but outliers, hence median filter is a better choice as median is not sensitive to outliers as opposed to gaussian which takes the average ")

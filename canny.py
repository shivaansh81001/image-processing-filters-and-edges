import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, img_as_ubyte,img_as_float, exposure,filters,feature
from scipy import signal, spatial
import sys

def Canny():
    def print_gaussian(original, gaussian):
        plt.figure(figsize=(5,10))
        plt.subplot(2,1,1),plt.imshow(original , cmap='gray'), plt.title("original")
        plt.subplot(2,1,2),plt.imshow(gaussian , cmap='gray'), plt.title("gaussian sigma = 1.0")
        plt.tight_layout()
        plt.show()
        
        
    def print_canny(target, my_image):
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1),plt.imshow(target , cmap='gray'), plt.title("Target image")
        plt.subplot(1,2,2),plt.imshow(my_image , cmap='gray'), plt.title("My Image")
        plt.tight_layout()
        plt.show()
    
    def read_image(filename):
        image = io.imread(filename, as_gray=True)
        img = img_as_ubyte(image)
        return img
    
    def canny(img, low_t, high_t, sig):
        return feature.canny(img, sigma = sig, low_threshold= low_t, high_threshold = high_t)
    
    
    original = read_image("assets/ex2.jpg")
    original_gaussian = filters.gaussian(original,sigma= 1)
    print_gaussian(original, original_gaussian)
    
    #print(np.max(original,axis=1))
    
    target = read_image("assets/canny_target.jpg")
    
    best_distance= 1e10
    
    best_params = [0,0,0] #low , high, sigma
    
    low_threshold = [50, 70, 90]
    high_threshold = [150, 170, 190]
    sigma = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
    
    for low in low_threshold:
        for high in high_threshold:
            for sig in sigma:
                canny_output = canny(original, low, high, sig)
                
                #print(canny_output.shape)
                #print(target.shape)
                this_dist = spatial.distance.cosine(canny_output.flatten().astype(float), target.flatten().astype(float))
                
                if this_dist < best_distance and (np.sum(canny_output>0.0)>0.0): 
                    best_distance = this_dist
                    best_params = [low,high,sig]
                    #print(best_params)
                    
    my_image = feature.canny(original, sigma = best_params[2],low_threshold= best_params[0], high_threshold = best_params[1])
                    
    print(f"best cosine distance: {best_distance}")
    print(f"best parameters : low = {best_params[0]}, high = {best_params[1]}, sigma = {best_params[2]}")
    
    print_canny(target, my_image)
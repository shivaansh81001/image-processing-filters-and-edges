import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, img_as_ubyte,img_as_float, exposure,filters,feature
from scipy import signal, spatial
import sys

def inpainting():
    def print_images(damaged,fixed):
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1),plt.imshow(damaged, cmap='gray'), plt.title("damaged")
        plt.subplot(1,2,2),plt.imshow(fixed, cmap='gray'), plt.title("fixed")
        plt.tight_layout()
        plt.show()
    
    def read_image(filename):
        image = io.imread(filename, as_gray=True)
        img = img_as_float(image)
        return img
    
    
    def inpainting(damaged,mask):
        
        output = damaged.copy()
        for i in range(50):    
            gaussian_applied = filters.gaussian(output, sigma = 1)
            output = np.where(mask==0,gaussian_applied,damaged)
            
        return output
    
    
    mask = read_image("assets/damage_mask.png")
    damaged = read_image("assets/damage_cameraman.png")
    
    fixed = inpainting(damaged,mask)
    
    print_images(damaged, fixed)

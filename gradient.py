import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, img_as_ubyte,img_as_float, exposure,filters,feature
from scipy import signal, spatial
import sys
def Gradient():
    def read_image(filename):
        image = io.imread(filename, as_gray = True)
        img= img_as_ubyte(image)
        return img
    
    def horizontal_gradient(original):
        kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        horizontal = signal.convolve2d(original, kernel, mode='same', boundary='fill', fillvalue=0)
        horizontal = horizontal/np.max(np.abs(horizontal)) *128
        return horizontal 
    
    def vertical_gradient(original):
        kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        vertical = signal.convolve2d(original, kernel, mode='same', boundary='fill', fillvalue=0)
        vertical = vertical/ np.max(np.abs(vertical))*128
        return vertical
    
    def gradient_magnitude(horizontal, vertical):
        magnitude = np.sqrt(np.square(horizontal)+np.square(vertical))
        return magnitude
    
    def print_images(original, horizontal, vertical, magnitude):
        plt.figure(figsize=(5,10))
        
        #GPT promt " how to add colorbar beside my image
        ax1 = plt.subplot(4, 1, 1)
        im1 = ax1.imshow(original, cmap='gray')
        plt.title("original")

        
        # Second subplot (horizontal)
        ax2 = plt.subplot(4, 1, 2)
        im2 = ax2.imshow(horizontal, cmap='gray')
        plt.title("horizontal")
        plt.colorbar(im2, ax=ax2)
        
        # Third subplot (vertical)
        ax3 = plt.subplot(4, 1, 3)
        im3 = ax3.imshow(vertical, cmap='gray')
        plt.title("vertical")
        plt.colorbar(im3, ax=ax3)
        
        # Fourth subplot (magnitude)
        ax4 = plt.subplot(4, 1, 4)
        im4 = ax4.imshow(magnitude, cmap='gray')
        plt.title("magnitude")
        plt.colorbar(im4, ax=ax4)
        plt.tight_layout()
        plt.show()
        
    
    original = read_image("assets/ex2.jpg")
    horizontal= horizontal_gradient(original)
    vertical = vertical_gradient(original)
    magnitude = gradient_magnitude(horizontal , vertical)
    
    print_images(original, horizontal, vertical, magnitude)
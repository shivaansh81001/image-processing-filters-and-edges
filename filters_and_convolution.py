import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, img_as_ubyte,img_as_float, exposure,filters,feature
from scipy import signal, spatial
import sys

def Filters():
    #print("imported filters")

    def read_image(filename):
        image = io.imread(filename, as_gray=True)
        img = img_as_ubyte(image)
        return img


    g_img = read_image("assets/moon.png")    
    plt.imshow(g_img,cmap='gray')
    plt.show()
    g_img_padded = np.pad(g_img, 1, mode = "constant")
    #print(g_img_padded.shape)   
    output_template = np.zeros(g_img.shape)

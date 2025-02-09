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
    
    def filtering(img,kernel,J):
        H,W = img.shape
        h= kernel.shape[0]//2
        w= kernel.shape[1]//2
        
        
        for i in range(h, H-h-1):
            for j in range(w, W-w-1):
                for k in range(-h,h+1):
                    for l in range(-w,w+1):
                        #print(f"i={i}, j={j},k={k},l={l}")
                        J[i,j] += img[i+k,j+l]*kernel[k+h,l+w]
        return J
    

    g_img = read_image("assets/moon.png")    
    #plt.imshow(g_img,cmap='gray')
    #plt.show()
    g_img_padded = np.pad(g_img, 1, mode = "constant")
    #print(g_img_padded.shape)   
    output_template = np.zeros(g_img.shape)

    ################################################################# laplacian #####################################################
    laplacian_output = np.copy(output_template)
    laplacian_filter = np.array([[0,-1,0],
                                 [-1,4,-1],
                                 [0,-1,0]])
    
    laplacian = filtering(g_img_padded,laplacian_filter,laplacian_output)
    #plt.imshow(laplacian,cmap='gray')
    #plt.show()
    #print(laplacian_filter)

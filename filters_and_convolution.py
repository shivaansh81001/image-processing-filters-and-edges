import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, img_as_ubyte,img_as_float, exposure,filters,feature
from scipy import signal, spatial
import sys

def Filters():
    #print("imported filters")

    def read_image(filename):
        '''input- filename
        output - read grayscale image
        reads images and return images in ubyte form'''
        image = io.imread(filename, as_gray=True)
        img = img_as_ubyte(image)
        return img
    

    def filtering(img,kernel,J):
        '''input- image and kernel
        output- resultant image
        convolves 1-D and 2-D kernel over the image to calculate the resluting image'''
        H,W = img.shape
        if len(kernel.shape) == 1:
            w=kernel.shape[0]//2  
           
            J=np.zeros_like(img,dtype=np.float64)
            
            temp = np.zeros_like(img, dtype=np.float64)
            for i in range(H):
                for j in range(w,W-w):
                    for k in range(-w,w+1):
                        #print(f"i={i}, j={j},k={k}")
                        temp[i,j] += img[i,j +k] *kernel[k+w]
        
            for i in range(w,H-w):
                for j in range(W):
                    for k in range(-w,w+ 1):
                        #print(f"i={i}, j={j},k={k}")
                        J[i,j]+=temp[i+k,j]*kernel[k+w]
        
        else:
            h= kernel.shape[0]//2
            w= kernel.shape[1]//2
            
            for i in range(h, H-h-1):
                for j in range(w, W-w-1):
                    for k in range(-h,h+1):
                        for l in range(-w,w+1):
                            #print(f"i={i}, j={j},k={k},l={l}")
                            J[i,j] += img[i+k,j+l]*kernel[k+h,l+w]
        return J
    
    def SSD(gaussian,b_part):
        '''input- gaussain and sobel
            output- sum of squared means
            calculates SSD - sum of sqaured means between two normalized images'''
        gaussian = gaussian.astype(np.float64) /np.max(gaussian)
        b_part = b_part.astype(np.float64) / np.max(b_part)
        
        #print(np.max(gaussian))
        #print(np.max(b_part))
        
        ssd = np.sum((gaussian.astype(np.float64) - b_part.astype(np.float64)) ** 2)
        ssd = ssd/2048
        return ssd

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
    ################################################################# Gaussian #####################################################
    
    gaussian_filter = (1/273)*np.array([[1,  4,   7,   4,  1],
                                        [4, 16, 26, 16, 4],
                                        [7, 26, 41, 26, 7],
                                        [4, 16, 26, 16, 4],
                                        [1,   4,   7,   4, 1]]).astype(np.float64)
    gaussian_output = np.copy(output_template)
    gaussian = filtering( g_img_padded, gaussian_filter , gaussian_output)
    #plt.imshow(gaussian ,cmap='gray')
    #plt.show()

    ################################################################# seperabel 1-D filter  #####################################################

    b_filter = (1/17)*np.array([1,  4,   7,   4,  1]).astype(np.float64)
    b_output= np.copy(output_template)
    
    b_part = filtering( g_img_padded, b_filter , b_output)
    b_part = b_part[1:-1,1:-1]
  
    #print(np.maximum(b_part,axis=1))
    print(b_part.shape)
    #plt.imshow(b_part ,cmap='gray')
    #plt.show()
    
    ################################################################# test kernel #####################################################
    
    q3_filter = np.array([[0,0,0,0,0],
                        [0,1,0,1,0],
                        [0,0,0,1,0]])
    q3_output = np.copy(output_template)
    q3 = filtering( g_img_padded, q3_filter , q3_output)
    #plt.imshow(q3 ,cmap='gray')
    #plt.show()

    ################################################################# test kernel 2 #####################################################
    q4_filter = np.array([[0,0,0],
                            [6,0,6],
                            [0,0,0]])
    q4_output = np.copy(output_template)
    q4 = filtering( g_img_padded, q4_filter , q4_output)
    #plt.imshow(q4 ,cmap='gray')
    #plt.show()


    ################################################################# SSD calculation #####################################################
    ssd=SSD(gaussian, b_part)
    print(f"SSD = {ssd}")


    ################################################################# Applying laplacian filter #####################################################
    laplacian_applied = np.maximum(g_img + laplacian,0)
    ################################################################# Applying the gaussian filter #####################################################
    gaussian_applied = np.maximum(laplacian_applied +  gaussian,0)

    ################################################################# Print results  #####################################################
    print_images(g_img,laplacian,gaussian,b_part , q3, q4)

    print_56(g_img, laplacian_applied, gaussian_applied)
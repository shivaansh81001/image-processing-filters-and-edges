import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, img_as_ubyte,img_as_float, exposure,filters,feature
from scipy import signal, spatial
import sys
from filters_and_convolution import Filters
from noise_removal import noise_removal
from inpainting import inpainting
from gradient import Gradient
from canny import Canny



def main():
    Filters()
    noise_removal()
    inpainting()
    Gradient()
    Canny()



if __name__=='__main__':
    main()
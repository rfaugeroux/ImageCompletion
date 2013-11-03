# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 18:25:45 2013

@author: Romain
"""

import math
import numpy as np
import skimage
from sklearn.feature_extraction import image
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt



""" A missing region is just an object who
    can tell whether a pixel is known or not. 
    For now, it is just a rectangle.
    """
class Missing_region:
    
    def __init__(self, I, J, height, width):
        self.i = I
        self.j = J
        self.h = height
        self.w = width
        
    def getHeight(self):
        return self.h
    
    def getWidth(self):
        return self.w
    
    def isMissing(self, pixelI,pixelJ):
        return self.i <= pixelI < self.i+self.h  \
        and self.j <= pixelJ < self.j+self.w
        
    def isMissingPatch(self, patch):
        return self.isMissing(patch.i, patch.j) \
        or self.isMissing(patch.i + patch.size-1, patch.j + patch.size-1)

def sq_norm(patch):
   return (patch**2).sum()

#a patch has to be a np.array 
def patch_dist(patch1, patch2): 
    return sq_norm(patch1-patch2)


"""Given a collection of patches, indexed by their position in the picture,
    this function finds the closest to the one given as an argument.
    Returns the offset between those two patches.
    """
def closest_patch(patch_i, patch_j, all_patches, unknown_region):    

    # We search in a rectangle whose size is proportional 
    # to the size of the missing region  
    search_rect_w = math.ceil(0.5*SEARCH_SPACE_FACTOR*unknown_region.getWidth())
    search_rect_h = math.ceil(0.5*SEARCH_SPACE_FACTOR*unknown_region.getHeight())

    # The threshold to have a minimum offset between the closest patches   
    threshold = 2*max(search_rect_w,search_rect_h)*THRESHOLD_FACTOR 
    
    # The search rectangle has to stay inside the image    
    startI = max(0, patch_i - search_rect_h)
    stopI = min(all_patches.shape[0], patch_i + search_rect_h)
    
    startJ = max(0, patch_j - search_rect_w)
    stopJ = min(all_patches.shape[1], patch_j + search_rect_w)
    
    dists = np.inf * np.ones((stopI - startI, stopJ - startJ))
    
    patch_compared = all_patches[patch_i, patch_j]    
    
    # Loop to compute the distance from our patch to all the othes
    # This is where the inefficiency comes from    
    for i in range(startI, stopI):
        for j in range(startJ, stopJ):
            norm = (startI + i - patch_i)**2 + (startJ + j - patch_j)**2
            # Tests if the two patches aren't too close
            if (norm > threshold):       ### WARNING, WE DON'T TEST YET FOR THE MISSING REGION
                current_patch = all_patches[startI + i, startJ + j]
                dists[i,j] = patch_dist(patch_compared, current_patch)
    
    min_flattened = dists.argmin()
    n_columns = stopJ - startJ

    # Computation of the minimum position coordinates    
    min_pos_i = min_flattened // n_columns
    min_pos_j = min_flattened % n_columns    
    
    best_offset_i = startI + min_pos_i - patch_i
    best_offset_j = startJ + min_pos_j - patch_j
        
    return (best_offset_i, best_offset_j) 


""" Computes the histogram of the best offsets distribution.
    """    
def offset_histogram(im, missing_region, patch_size):
    
    (im_height, im_width) = im.shape   
    
    all_patches = image.extract_patches_2d(im, (patch_size, patch_size))
    nb_rows = im_height - patch_size + 1
    nb_cols = im_width - patch_size + 1 
    
    # The array of patches is reshaped to have each patch indexed by its
    # coordinates in the image    
    all_patches = all_patches.reshape( nb_rows, nb_cols, patch_size, patch_size)

    #The size of the histogram is determined by the maximum offsets possible
    hist_height = 2*min(im_height, SEARCH_SPACE_FACTOR*missing_region.getHeight())    
    hist_width = 2*min(im_width, SEARCH_SPACE_FACTOR*missing_region.getWidth())
    
    hist = np.zeros((hist_height, hist_width))
    
    for i in range(nb_rows):
        for j in range(nb_cols):
            (x, y) = closest_patch(i, j, all_patches, missing_region)
            # The zero of the offsets is the center of the histogram            
            hist[hist_height//2 + x, hist_width//2 + y] += 1
            
    return hist
    

    


###################     GLOBAL VARIABLES     #########################


PATCH_SIZE = 4
NB_PEAKS = 60
SEARCH_SPACE_FACTOR = 3
THRESHOLD_FACTOR = 1/15



#######################    TEST PART     #############################


""" Initialization of the test picture """
picture = np.zeros((32, 32))
col = 2*np.arange(32)
for i in range(8):
    picture[:,4*i] = col

""" Noise is added to have no exact matches """
noise = np.random.normal(size=picture.shape)
picture+=noise
    

""" the unknown region, whose size is used to bound the distance between two
    matched patches """
missing = Missing_region(0,0,20,20)
    

hist = offset_histogram(picture, missing, PATCH_SIZE)


""" The histogram is filtered to keep only the interesting peaks """ 
#filtered_hist = ndimage.gaussian_filter(hist, math.sqrt(2))


#plt.imshow(hist, cmap='gray', interpolation='nearest')  
#plt.imshow(picture, cmap='gray', interpolation = 'nearest')







    
    
            
            
    
    
    

   

         
    

    
    

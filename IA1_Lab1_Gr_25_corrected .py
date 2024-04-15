#!/usr/bin/env python
# coding: utf-8

# 

# # Image analysis 1 ()
# ## Lab 1: Pre-processing

# | Group __25__ | Name  | Matr.-nr. |
# |-|-|-|
# |   Member 1  | Vimal Chawda | 10025862 |
# |   Member 2  | 
# |   Member 3  | name3 | 12345 |

# Required __imports__ for this lab. Don't forget to run the following cell (which may __take up some minutes__ because some functions are getting compiled).

# In[1]:


import lab                         # given functions
from numba import jit              # faster computations
import numpy as np                 # numerical computations
import matplotlib.pyplot as plt    # used to plot the histogram in Ex. 2.3 
import math 


# ## Exercise 1: Image smoothing
# ### Exercise 1.1: Gaussian filter implementation
# __Implement__ the function in the next cell that returns a Gaussian filter matrix $H_g$ of size $n \times n$ with a given standard deviation $\sigma$
# 
# $$ H_g(i,j) = \dfrac{1}{2\pi\sigma^2}\cdot e^{-\dfrac{i^2+j^2}{2\sigma^2}} $$
# 
# where $H_g(0,0)$ is the __center__ of the filter matrix! 
# 
# The filter size $n$ should be at least $k \cdot \sigma$ with $k = 6$. Also $n$ should be positive and odd.

# In[4]:


def gauss_filter(sigma, k = 6):
    assert sigma > 0.0, "Std. deviation must be positive!"
    n = int(k * sigma) # filter width and height should be at least 6 * sigma
    if n % 2 == 0:     # if n is even, ..
        n += 1         # .. we add one to make it odd

    H = np.zeros((n, n), dtype=np.float64) # initialize the filter matrix H with zeros
    
    # YOUR CODE GOES HERE 
    for i in range(n):
        for j in range(n):
            
            i_2 = (i-(n-1)/2)**2      # the window is the center H(0,0)
            j_2 = (j-(n-1)/2)**2
            sigma_2 = sigma**2
            exception = math.exp(-(i_2+j_2)/(2*sigma_2))    #we can do it by np also instead of math
            H[i,j] = (1/(2*math.pi*sigma_2))*exception

    return H


# <font color='green' size='5'><b>OK</b></font>
# 
# <font color='orange'><b></b></font>

# ### Exercise 1.2: Median filter implementation
# __Implement__ the function below that applies a median filter to an image (one or three channels!). The first argument is the input image $I$, the second is the mask width $n$. The median filter should be based on a square $n\times n$ mask.

# In[5]:


@jit(nopython=True, cache=True) # uncomment to improve computation speed
def median_filter(I, n):
    assert n > 0, "Context width must be positive!"
    if n % 2 == 0:        # if n is even, ..
        n += 1            # .. we add one to make it odd
        
    R = np.zeros_like(I)  # initialize the result image R with zeros (shape and type of I)
        
    # YOUR CODE GOES HERE 
    image = lab.extend_same(I, int((n-1)/2)) #'lab.extend_same(I, m)' to extend the i/p images I by m pixels  respective to side 
    for k in range(R.shape[2]):   # 3loop for color 
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                 X = image[i:i+n,j:j+n,k]
                 R[i,j,k] = np.median(X)               
   
    # hint 1: recall the implementation of a convolution in the tutorial
    # hint 2: np.median(X) will return the median value of a matrix X
    # hint 3: use 'lab.extend_same(I, m)' to extend the input image I by m pixels on each side
    #         (so that the returned array R has the same shape as the input array I)

    return R


# <font color='green' size='5'><b>OK</b></font>
# 
# <font color='orange'><b></b></font>

# ### Exercise 1.3: Comparison and evaluation
# The code in the following cell adds noise to the example image (white noise left and salt peeper noise right) and applies the Gaussian filter and the median filter to the noisy image. __Select appropriate filter parameters__ to reduce the noise. __Write__ a brief discussion, which contains the following aspects:
# 1. Main idea of the implemented blurring techniques (Gaussian and median).
#    - Comparison of gauss and median filtering results with respect to the noise type.
# 2. Improvements and drawbacks of the Gaussian filter with respect to a moving average filter.
# 3. Influence of the factor $k$ on the results of the gauss filter. Run the example at least with $k = 1$ and $k = 12$, document and explain your observations.

# In[6]:


I = lab.imread3D('images/lena_color.jpg')        # load example image
I_noise = lab.add_mixed_noise(I, 40, 0.15)       # add noise to the image

# PARAMETERS:
sigma_gauss =  3        # std. dev. of gaussian filter [> 1.0]  # tried with different values
n_median    =  5        # size of median window        [>= 3]

H_gauss = gauss_filter(sigma_gauss)              # create a gaussian filter matrix
I_gauss = lab.convolution(I_noise, H_gauss)      # convolve image with gaussian filter
I_median = median_filter(I_noise, n_median)      # applying median filter

lab.imshow3D(I, I_noise, I_gauss)                # display the results
print('left to right: orig. image | with noise | gauss filtered')
lab.imshow3D(I, I_noise, I_median)
print('left to right: orig. image | with noise | median filtered')


# #### Discussion:
# 
# #### 1-Gaussian Filter
# GaussianFilter is a filter commonly used in image processing for smoothing, reducing noise, and computing derivatives of an image. It is not particularly effective at removing salt and pepper noise. Gaussian filtering is not edge preserving.
# Gaussian filtering is linear, meaning it replaces each pixel by a linear combination of its neighbors (in this case with weights specified by a Gaussian matrix). 
# 
# #### 2- Median Filter 
# MedianFilter is a nonlinear filter commonly used to locally smooth data and diminish noise, it is also well known to remove salt-and-pepper noise from images.
# MedianFilter replaces each pixel by a pixel in its neighborhood that has the median total intensity, averaged over all channels.
# 
# ##### 3
# - You can emulate an approximation of a Gaussian filter by combining several moving average ones of different lengths.
# 
# ##### 4
# - In our case we have tried different cases and we found that the window size 3 which does not ample the  results, so we have to increase the size of window to 5, which provide better results as compared to previous one with increase in the computation of time. Hence, if we increase the factor k then subsquently the level of smoothing of an image is increases . <font color='red'><b>[Not correct]</b></font>

# ## Exercise 2: Histograms
# ### Exercise 2.1: Histogram normalization
# __Implement__ the function below that applies a histogram normalization on a __single channel image__ $I$. The parameter $s$ denotes the outlier fraction.

# In[2]:


# @jit(nopython=True, cache=True) # uncomment to improve computation speed
def histogram_normalization(I, s):
    R = np.zeros_like(I) # initialize the result image R with zeros (shape and type of I)
    
    # YOUR CODE GOES HERE
    
    # step 1: search for the lower/upper thresholds for which the number of  
    #         pixels with a smaller/higher gray value exceeds the quantile
    #
    shapeI = np.shape(I)
    sizeI = shapeI[0] * shapeI[1]
    if s>0:
        discardedPixels = math.floor((sizeI*s)/2)
        F=I.flatten()
        F=sorted(F)
        for k in range(discardedPixels-1):
            minGrayValThreshold = F[discardedPixels]
            maxGrayValThreshold = F[sizeI-discardedPixels]
    else:   # in case s=0
        minGrayValThreshold=0
        maxGrayValThreshold=255
        
    # step 2: rescale/shift all pixel values according to these thresholds 
    #         so that the lower threshold is mapped to 0 and the upper to 255
    #
    for i in range(shapeI[0]-1):
        for j in range(shapeI[1]-1):
            R[i,j]= (I[i,j] - minGrayValThreshold)*((255)/(maxGrayValThreshold-minGrayValThreshold))
    
    
    # step 3: clip the pixel values so no pixel has a value g < 0 or g > 255
    for i in range(shapeI[0]-1):
        for j in range(shapeI[1]-1):
            if R[i,j] > 255 : 
                R[i,j]= 255
            if R[i,j] < 0 :
                R[i,j]= 0
    
    return R
 


# <font color='green' size='5'><b>OK</b></font>
# 
# <font color='orange'><b></b></font>

# ### Exercise 2.2: Histogram equalization
# __Implement__ the function below that performs a histogram equalization on a __single channel image__ $I$.

# In[3]:


# @jit(nopython=True, cache=True) # uncomment to improve computation speed
def histogram_equalization(I):
    R = np.zeros_like(I) # initialize the result image R with zeros (shape and type of I)
    
    # YOUR CODE GOES HERE
    
    # step 1: cumpute the cumulated histogram and 
    #         normalize it to the range [0-255]
    #
    nbr_bins=256
    imhist,bins = np.histogram(I.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution fnct
    cdf = 255 * cdf / cdf[-1] # normalize    
    
    # step 2: use this histogram as a mapping fnct
    #         and allocate each pixel wrt corresponding values
    R = np.interp(I.flatten(),bins[:-1],cdf)
    R = R.reshape(I.shape)
    
    return R


# <font color='green' size='5'><b>OK</b></font>
# 
# <font color='orange'><b></b></font>

# ### Exercise 2.3: Comparison and evaluation
# The following cell will evaluate the implemented functions. __Try different outlier fractions__ and __discuss the results__. Emphasize on the following aspects:
# 1. Basic idea of each method (you may refer to the plots).
#   - Similarities and differences between the two methods.
#   - Advantages and drawbacks of each method.
# 2. How could the methods be applied to multi-channel images?

# In[4]:


I = lab.imread3D('images/monkey.jpg')

# PARAMETERS:
s =0.04         # outlier fraction for normalization [0 - 1]

I_norm = histogram_normalization(I, s)            # apply histogram normalization
I_equl = histogram_equalization(I)                # apply histogram equalization

lab.imshow3D(I, I_norm, I_equl)                   # display all results
print('left to right: original image | normalized | equalized' )

# PLOTS:
get_ipython().run_line_magic('matplotlib', 'inline')
# next line defines size of plots (E.G. FIRST VALUE CHANGES WIDTH OF PLOTS)
plt.rcParams["figure.figsize"] = [18, 6]

plt.figure('hists')         
plt.subplot(121)
bins = plt.hist(I.ravel(),      256, (0,255), histtype='bar', label='original',   rwidth=1.0)
bins = plt.hist(I_norm.ravel(), 256, (0,255), histtype='bar', label='normalized', rwidth=0.5)
lgnd = plt.legend()
titl = plt.title('Histogram')
plt.subplot(122)
bins = plt.hist(I.ravel(),      256, (0,255), histtype='step', label='original',  cumulative = True)
bins = plt.hist(I_equl.ravel(), 256, (0,255), histtype='step', label='equalized', cumulative = True)
lgnd = plt.legend(loc=2)
titl = plt.title('Cumulative Histogram')


# #### Discussion:
# ### Histogram Equalization 
# is a computer image processing technique used to improve contrast in images . It accomplishes this by effectively spreading out the most frequent intensity values  (stretching out the intensity range of the image).It provides better quality of images without loss of any information.
# 
# ##### Advantages of histogram equalization:
# - if the histogram equalization function is known, then the original histogram can be recovered.
# - adjust the image to make it easier to analyze or improve visual quality.
# - simple and effective
# 
# ##### Disadvantages of histogram equalization:
# - It may increase the contrast of background noise, while decreasing the usable signal
# 
# ### Histogram Normalization 
# is a common technique that is used to enhance fine detail within an image.  Each column in the cumulative histogram is computed as the sum of all the image intensity histogram values up to and including that grey level, and then it is scaled so that the final value is 1.0. Normalization is sometimes called contrast stretching or histogram stretching. 
# 
# ##### Advantages of histogram normalization:
# - simple and effective
# - streches the gray value range
# - simple point operation used to improve images with poor contrast.
# - The normalized histogram is the probability distribution occurrences of i_k intensity values in the image
# 
# ##### Disadvantages of histogram normalization:
# - histogram normalization of digital image usually involeves loss of information
# - there are normally gaps in grey level sequence after normalization 
# 
# For equalizing the histogram, we need to compute the histogram and then normalize it into a probability distribution. For normalization, we just need to divide the frequency of each pixel intesity value by the total number of pixels present in the image.
# 
# 
# ### Methods of histogram equalization on multi-channel image 
# - Independent histogram equalization based on color channel
# - Histogram equalization based on average value of color channel
# - Intensity component equalization based on HSI color space
# 
# 

# ## Exercise 3: Image gradients
# ### Exercise 3.1: Derivative of gaussian
# __Implement__ the function below, that creates the derivative of gaussian filter matrices in x and y direction with given standard deviation $\sigma$. 
# Recall that
# 
# $$ H_x(i,j) = -\dfrac{i}{2\pi\sigma^4}\cdot e^{-\dfrac{i^2+j^2}{2\sigma^2}} \hspace{2cm}\&\hspace{2cm} H_y(i,j) = -\dfrac{j}{2\pi\sigma^4}\cdot e^{-\dfrac{i^2+j^2}{2\sigma^2}}$$
# 
# where $H_x(0,0)$ and $H_y(0,0)$ are the __centers__ of the filter matrices! 
# 
# Note that the  $i$-axis is parallel to the image $x$-axis (pointing down) and the $j$-axis is parallel to the image $y$-axis (pointing right).

# In[10]:


def derivative_of_gaussian(sigma):
    n = int(6 * sigma) # filter width and height should be at least 6 * sigma
    if n % 2 == 0:     # if n is even, ..
        n += 1         # .. we add one to make it odd

    Hx = np.zeros((n, n), dtype=np.float64) # initialize filter matrices with zeros
    Hy = np.zeros((n, n), dtype=np.float64) 
    
    # YOUR CODE GOES HERE
    for i in range(n):
        for j in range(n):
            a = i-(n+1)/2;
            b = j-(n+1)/2;
            Hx[i,j] = -(a/2/np.pi/np.power(sigma,4))*(np.exp(-1*(np.square(a)+np.square(b))/2/np.square(sigma)))
            Hy[i,j] = -(b/2/np.pi/np.power(sigma,4))*(np.exp(-1*(np.square(a)+np.square(b))/2/np.square(sigma)))
                
    return Hx, Hy


# <font color='green' size='5'><b>OK</b></font>
# 
# <font color='orange'><b></b></font>

# ### Exercise 3.2: Laplacian of gaussian
# __Implement__ the function below, that creates the laplacian of gaussian filter matrix $H_L$ with given standard deviation $\sigma$. Recall that
# 
# $$ H_L(i,j) = \dfrac{i^2 + j^2 - 2\sigma^2}{2\pi\sigma^6}\cdot e^{-\dfrac{i^2+j^2}{2\sigma^2}}$$
# 
# where $H_L(0,0)$ is the __center__ of the filter matrix!

# In[11]:


def laplacian_of_gaussian(sigma):
    n = int(6 * sigma) # filter width and height should be at least 6 * sigma
    if n % 2 == 0:     # if n is even, ..
        n += 1         # .. we add one to make it odd
        
    H = np.zeros((n, n), dtype=np.float64) # initialize the filter with zeros
    
    # YOUR CODE GOES HERE
    for i in range(n):
        for j in range(n):
            a = i-(n+1)/2;
            b = j-(n+1)/2;
            H[i,j] = ((np.square(a)+np.square(b)-2*np.square(sigma))/2/np.pi/np.power(sigma,6))*(np.exp(-1*(np.square(a)+np.square(b))/2/np.square(sigma)))

    return H    



# <font color='green' size='5'><b>OK</b></font>
# 
# <font color='orange'><b></b></font>

# ### Exercise 3.3: Comparison and evaluation
# The next cell visualizes the filters (use this to validate your implementations)
# 
# Left to right: $H_x$ , $H_y$ , $H_L$

# In[12]:


fig, ax = plt.subplots(1,3) # Caution, figsize will also influence positions.
im1 = ax[0].imshow(derivative_of_gaussian(6)[0], vmin = -0.00015, vmax =0.00015)
im1 = ax[1].imshow(derivative_of_gaussian(6)[1], vmin = -0.00015, vmax =0.00015)
im2 = ax[2].imshow(laplacian_of_gaussian(6), vmin = -0.00015, vmax =0.00015)
fig.colorbar(im1, ax=ax)
plt.show()


# The following cell will evaluate and display the results of your implementations. Note that the results are normalized to a range between 0 and 255. __Try different standard deviations__ for the filters and __discuss the results__. Emphasize on the following aspects:
# 1. Influence of the standard deviation $\sigma$ on the results.
# 2. Alternative operations / filters to compute the first derivative of an image.

# In[13]:


I = lab.imread3D('images/monkey.jpg')

# PARAMETERS:
sigma_deriv =2.1       # std. dev. of derivative of gaussian filter [> 1.0]
sigma_L     =1.7       # std. dev. of laplacian of gaussian filter [> 1.0]

Hx, Hy = derivative_of_gaussian(sigma_deriv)  # create filter matrices
HL     = laplacian_of_gaussian(sigma_L)

Ix = lab.convolution(I, Hx)                   # apply filter matrices
Iy = lab.convolution(I, Hy) 
IL = lab.convolution(I, HL)

Ix_n = lab.normalize(Ix)                      # normalize results to range [0 - 255]
Iy_n = lab.normalize(Iy)
IL_n = lab.normalize(IL)

lab.imshow3D(I, Ix_n, Iy_n, IL_n)             # display all results
print('left to right: orig. image | deriv. x | deriv. y | laplacian')


# #### Discussion
# 
# The Influence of standard Deviation 
# 
# If the sigma[standard deviation] increses then resultant image become blurrier which also shows the noise is reducing and vice versa. At below we have used sigma value as 2. It shows in the resultant image obtained. 
# 
# Alternative Operations
# 
# We have many approaches are as :-
# 1- Sobel Operators
# 2-Roberts Cross
# 3-Differential
# 4-Deriche edge detector
# 

# <font color='green' size='5'><b>OK</b></font>
# 
# <font color='orange'><b></b></font>

# ### Exercise 3.4: Gradient magnitude and orientation
# Use your implementation of the derivative of Gaussian to __compute the magnitude and orientations of the gradients__. The magnitude $M(x, y)$ and gradient direction $D(x, y)$ of a pixel at $(x, y)$ are defined as
# $$M(x,y) = \sqrt{I_x(x,y)^2 + I_y(x,y)^2} $$ and
# $$D(x,y) = atan2(I_y(x,y), I_x(x,y)) .$$
# 
# where $I_x(x,y)$ and $I_y(x,y)$ are the image derivatives in $x$ and $y$ direction (e.g. from convolution with the derivative of Gaussian filter). Check the results for plausibility!

# In[14]:


I = lab.imread3D('images/lena.jpg')

sigma = 2.0                 # we are using sigma value as 2

Hx, Hy = derivative_of_gaussian(sigma)
Ix = lab.convolution(I, Hx)   
Iy = lab.convolution(I, Hy) 

# YOUR CODE GOES HERE
M =  np.sqrt(np.square(Ix) + np.square(Iy))
D =  np.arctan2(Iy,Ix)

M_n = lab.normalize(M)
D_n = lab.normalize(D)
P   = lab.prettify_gradients(M, D)  # directions mapped to hue and magnitudes to value

lab.imshow3D(I, M_n, D_n, P)        # display all results
print('left to right: original | gradient magnitudes | gradient directions | color visualization')


# <font color='green' size='5'><b>OK</b></font>
# 
# <font color='orange'><b></b></font>

# In[ ]:





# In[ ]:





# In[ ]:





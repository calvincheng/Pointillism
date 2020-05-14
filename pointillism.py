import math
import random
import numpy as np
from scipy import signal
from scipy import ndimage
from matplotlib import pyplot as plt
import os

import argparse
parser = argparse.ArgumentParser(description='Generate pointillist art with Unicode braille.')
parser.add_argument('-b', '--blur', 
                    type = float, 
                    metavar = '', 
                    default = 1,
                    help = 'Gaussian blur standard deviation')
parser.add_argument('-nt', '--no-thin',
                    default = True,
                    action = 'store_false',
                    help = 'Disable edge thinning')
parser.add_argument('-lr', '--low-ratio', 
                    type = float, 
                    metavar = '', 
                    default = 0.1,
                    help = 'Canny low threshold')
parser.add_argument('-hr', '--high-ratio', 
                    type = float, 
                    metavar = '', 
                    default = 0.2,
                    help = 'Canny high threshold')
parser.add_argument('--width', 
                    type = float, 
                    metavar = '', 
                    default = None,
                    help = 'Output width')
parser.add_argument('--height', 
                    type = float, 
                    metavar = '', 
                    default = None,
                    help = 'Output height')
parser.add_argument('--fit',
                    default = False,
                    action = 'store_true',
                    help = 'Fit printed result to terminal screen')
parser.add_argument('-o', '--output',
                    type = str,
                    metavar = '',
                    default = None,
                    help = 'Filename for .txt file to save result')
parser.add_argument('PATH',
                    type = str, 
                    help = 'Image path')
args = parser.parse_args()


def getGauss(x, y, mu, sigma):
    '''Generates Guassian function G(x, y)
    :mu: Average
    :sigma: Standard deviation
    '''
    coeff1 = 1 / math.sqrt(2 * math.pi * sigma**2)
    coeff2 = math.exp(-(x**2 + y**2) / (2 * sigma**2))
    return coeff1 * coeff2

def generate_gaussian_kernel(sigma = 1, n = 5):
    '''Generates a n-by-n zero-mean Gaussian kernel
    :sigma: Standard deviation used in the Gaussian function
    :n: Edge length
    '''
    return [[getGauss(x - n//2, y - n//2, 0, sigma) for x in range(n)] for y in range(n)]

def gaussian_filter(img, sigma = 1):
    '''Applies Gaussian filter to image
    :img: Input image
    :sigma: Standard deviation used in the Gaussian kernel – higher is blurrier
    '''
    kernel = generate_gaussian_kernel(sigma)
    result = ndimage.convolve(img, kernel, mode='nearest') # signal.convolve(img, kernel, mode='same')
    return result

def sobel(img):
    '''Outputs sobel magnitude and direction (in degrees)
    :img: Input image
    '''
    SOBEL_X = np.array([[ 1,  0, -1],
                        [ 2,  0, -2],
                        [ 1,  0, -1]])

    SOBEL_Y = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])

    G_x = ndimage.convolve(img, SOBEL_X, mode='nearest') # signal.convolve2d(img, SOBEL_Y, mode='same')
    G_y = ndimage.convolve(img, SOBEL_Y, mode='nearest') # signal.convolve2d(img, SOBEL_X, mode='same')

    G = np.sqrt(G_x**2 + G_y**2)
    theta = np.arctan2(G_y, G_x) * (180 / math.pi)
    return G, theta

def non_maximum_suppression(G, theta):
    '''Performs non-maximum suppression (edge thinning)
    :G: Input sobel magnitude array
    :theta: Input sobel direction array
    '''
    m, n = G.shape
    result = np.zeros((m, n))

    # Round theta to nearest 45 degree angle
    round_to = 45
    theta = (theta + round_to / 2) // round_to * round_to

    for i in range(m):
        for j in range(n):
            try:
                direction = theta[i][j]
                if direction == 0 or direction == 180 or direction == -180 :
                    nbr1 = G[i][j-1]
                    nbr2 = G[i][j+1]
                elif direction == 45 or direction == -135:
                    nbr1 = G[i-1][j-1]
                    nbr2 = G[i+1][j+1]
                elif direction == 90 or direction == -90:
                    nbr1 = G[i-1][j]
                    nbr2 = G[i+1][j]
                elif direction == 135 or direction == -45:
                    nbr1 = G[i-1][j+1] # G[i+1][j+1] ??
                    nbr2 = G[i+1][j-1]
                
                if G[i][j] > nbr1 and G[i][j] > nbr2:
                    result[i][j] = G[i][j]
                else:
                    result[i][j] = 0
                    
            except IndexError as e:
                pass

    return result

def threshold(img, low_ratio = 0.5, high_ratio = 0.2):
    '''High-low thresholding. Splits pixel into strong, weak and zero groups.
    :img: Input image
    :low_ratio: Low threshold ratio
    :high_ratio: High threshold ratio
    '''
    high = img.max() * high_ratio
    low = high * low_ratio
    
    m, n = img.shape
    res = np.zeros((m, n))
    
    STRONG = 255
    WEAK = 50
    
    strongs = set()
    weaks = []
    
    for i in range(m):
        for j in range(n):
            if img[i][j] > high:
                strongs.add((i, j))
                res[i][j] = STRONG
            elif low < img[i][j] <= high:
                weaks.append((i, j))
                res[i][j] = WEAK

    return res, strongs, weaks

def hysteresis(img, weaks, strongs):
    '''Hysteresis thresholding - Keeps weak pixels connected to strong ones
    :img: Input image
    :weaks: List of tuples of weak pixels positions
    :strongs: Set of tuples of strong positions
    '''
    m, n = img.shape
    res = np.zeros((m, n))
    
    strong = img.max()
    for i, j in strongs:
        res[i][j] = strong
    
    # Downward pass
    for i, j in weaks:
        nbrs = [(i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1), 
                (i, j-1), (i-1, j-1), (i-1, j), (i-1, j+1)]
        for nbr in nbrs:
            if nbr in strongs:
                res[i][j] = strong
                strongs.add((i, j))
                break
                
    # Upward pass
    for i, j in reversed(weaks):
        nbrs = [(i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1), 
                (i, j-1), (i-1, j-1), (i-1, j), (i-1, j+1)]
        for nbr in nbrs:
            if nbr in strongs:
                res[i][j] = strong
                strongs.add((i, j))
                break
                
    return res

def canny(img, blur_SD = 1, low = 0.1, high = 0.2, thin = True):
    '''Performs Canny edge detection
    :blur_SD: Standard deviation for Gaussian blur
    :low: low cutoff ratio (between 0 and 1)
    :high: high cutoff ratio (between 0 and 1)
    :thin: Enables edge thinning if True
    '''

    result = gaussian_filter(img, blur_SD)
    result, theta = sobel(result)
    if thin: result = non_maximum_suppression(result, theta)
    result, strongs, weaks = threshold(result, low, high)
    result = hysteresis(result, weaks, strongs)
    return result

def braille(img, width = None, height = None, fit = False):
    '''Generates braille output matrix of img
    :width: Output width
    :height: Output height
    :fit: Output fits within terminal window size if True
    '''

    BRAILLE = (( 1,   8),
               ( 2,  16),
               ( 4,  32),
               (64, 128))
    BLANK = 10240

    m, n = img.shape

    if fit:
        # Fit result to terminal window
        terminal_size = os.get_terminal_size();
        if (terminal_size.columns * 2) / ((terminal_size.lines - 1) * 4) < 1:
            # Portrait A-R
            width = terminal_size.columns * 2
            height = m * (width / n)
        else:
            # Landscape A-R
            height = (terminal_size.lines - 1) * 4
            width = n * (height / m)
    else:
        # Default to original image size if no dimensions are given
        if width is None and height is None:
            height, width = m, n

        # If only one dimension specified, retain aspect ratio
        elif width is None:
            height, width = height, n * (height / m)
        elif height is None:
            height, width = m * (width / n), width
        
    dy, dx = m / height, n / width

    res = np.zeros((int(round(height / 4)), int(round(width / 2))), dtype='int')
    for i in range(0, int(round(height)), 4):
        for j in range(0, int(round(width)), 2):
            block = BLANK
            for ii in range(i, i + 4):
                for jj in range(j, j + 2):
                    try:
                        if img[int(round(ii * dy))][int(round(jj * dx))] > 0.5:
                            block += BRAILLE[ii-i][jj-j]
                    except IndexError as e:
                            pass
                try:
                    res[i // 4][j // 2] = block
                except IndexError as e:
                    pass
    return res

def output(B, filename = None):
    '''Prints Braille matrix B
    :filename: If provided, outputs result in a .txt file named `filename` instead
    '''
    m, n = B.shape

    if filename:
        if os.path.isfile('{}.txt'.format(filename)):
            response = None
            while not (response == 'y' or response == 'n'):
                print('File named {}.txt already exists. Overwrite? [y / n]'.format(filename), end = ' ')
                response = input()
                if response == 'n':
                    print('Output cancelled.')
                    return
                elif response == 'y':
                    # Delete file contents
                    open('{}.txt'.format(filename), 'w').close()

        with open('{}.txt'.format(filename), 'a') as f:
            for i in range(m):
                for j in range(n):
                    print(chr(B[i][j]), end = '', file = f)
                print(file = f)

        print('Output written to {}.txt'.format(filename))

    else:
        os.system('clear')
        for i in range(m):
            for j in range(n):
                print(chr(B[i][j]), end = '')
            print()


# Driver code
if __name__ == '__main__':
    if args.PATH[-4:] != '.png':
        print('ERROR: Image must be in .png format')
        exit(1)

    try:
        img = plt.imread(args.PATH)
    except:
        print('ERROR: Invalid file path. Please try again.')
        exit(1)

    if np.ndim(img) > 2:
        # Convert RGB image to greyscale if needed
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    edge = canny(img, args.blur, args.low_ratio, args.high_ratio, args.no_thin)
    B = braille(edge, args.width, args.height, args.fit)
    output(B, args.output)

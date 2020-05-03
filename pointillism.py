import math
import random
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import os

import argparse
parser = argparse.ArgumentParser(description='Generate Braille pointillist art.')
parser.add_argument('-b', '--blur', 
                    type = float, 
                    metavar = '', 
                    default = 1,
                    help = 'Gaussian blur standard deviation')
parser.add_argument('-nt', '--no-thin',
                    default = True,
                    action = 'store_false',
                    help = 'Disable edge thinning')
parser.add_argument('--low-ratio', 
                    type = float, 
                    metavar = '', 
                    default = 0.1,
                    help = 'Canny low threshold')
parser.add_argument('--high-ratio', 
                    type = float, 
                    metavar = '', 
                    default = 0.2,
                    help = 'Canny high threshold')
parser.add_argument('--dx', 
                    type = float, 
                    metavar = '', 
                    default = 1,
                    help = 'Horizontal resolution')
parser.add_argument('--dy', 
                    type = float, 
                    metavar = '', 
                    default = 1,
                    help = 'Vertical resolution')
parser.add_argument('-o', '--output',
                    type = str,
                    metavar = '',
                    default = None,
                    help = 'Output filename')
parser.add_argument('PATH',
                    type = str, 
                    help = 'Image path')
args = parser.parse_args()


def getGauss(x, y, mu, sigma):
    '''Generates G(x, y) of a Gaussian function of 
    average mu and standard deviation sigma
    '''
    coeff1 = 1 / math.sqrt(2 * math.pi * sigma**2)
    coeff2 = math.exp(-(x**2 + y**2) / (2 * sigma**2))
    return coeff1 * coeff2

def generate_gaussian_kernel(sigma = 1, n = 5):
    '''Generates a n-by-n zero-mean Gaussian kernel
    :sigma: Standard deviation used in the Gaussian function
    '''
    return [[getGauss(x - n//2, y - n//2, 0, sigma) for x in range(n)] for y in range(n)]

def gaussian_filter(img, sigma = 1):
    '''Applies Gaussian filter to image
    :img: Input image
    :sigma: Standard deviation used in the Gaussian kernel – higher is blurrier
    '''
    kernel = generate_gaussian_kernel(sigma)
    result = signal.convolve(img, kernel, mode='same')
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
    G_x = signal.convolve(img, SOBEL_X, mode='same')
    G_y = signal.convolve(img, SOBEL_Y, mode='same')
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

def threshold(img, low_ratio = 0.3, high_ratio = 0.2):
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
    
    # First pass downwards
    for i, j in weaks:
        nbrs = [(i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1), 
                (i, j-1), (i-1, j-1), (i-1, j), (i-1, j+1)]
        for nbr in nbrs:
            if nbr in strongs:
                res[i][j] = strong
                strongs.add((i, j))
                break
                
    # Second pass upwards
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
    '''Performs Canny edge detection'''
    result = gaussian_filter(img, blur_SD)
    result, theta = sobel(result)
    if thin:
        result = non_maximum_suppression(result, theta)
    result, strongs, weaks = threshold(result, low, high)
    result = hysteresis(result, weaks, strongs)
    return result

def braille(img, dx = 1, dy = 1):
    BRAILLE = (( 1,   8),
               ( 2,  16),
               ( 4,  32),
               (64, 128))
    BLANK = 10240
    m, n = img.shape
    res = np.zeros((int(round(m / dy / 4)), int(round(n / dx / 2))), dtype='int')
    for i in range(0, int(round(m / dy)), 4):
        for j in range(0, int(round(n / dx)), 2):
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
    '''Prints Braille matrix B'''
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


if __name__ == '__main__':
    try:
        img = plt.imread(args.PATH)
    except:
        print('ERROR: Invalid file path. Please try again.')
        exit(1)
    img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]) # Flatten to m by n BW image
    edge = canny(img, args.blur, args.low_ratio, args.high_ratio, args.no_thin)
    B = braille(edge, args.dx, args.dy)
    output(B, args.output)

<img src="./images/moon_banner.png" alt="banner" style="zoom:6%;" />


# Pointillism

An experiment with edge detection and unicode braille to generate pointillism art.  
Inspired by the artist [Stanislava Pinchuk](https://stanislavapinchuk.com/) (also known as m-i-s-o).




## Getting started

Run the program via a console with

```
python3 pointillism.py PATH/TO/image.png
```



Additional arguments can be specified to refine the result. For example, running

```
python3 pointillism.py PATH/TO/image.png --height=240 width==480 -o=my_result
```
generates a 480 by 240 pixel image, and outputs the result to `my_result.txt` within the working directory.



To see a list of available arguments, running the program with the `-h` flag gives the following message:

```
usage: pointillism.py [-h] [-b] [-nt] [-lr] [-hr] [--width] [--height] [--fit]
                      [-o]
                      PATH

Generate pointillist art with Unicode braille.

positional arguments:
  PATH                 Image path

optional arguments:
  -h, --help           show this help message and exit
  -b , --blur          Gaussian blur standard deviation
  -nt, --no-thin       Disable edge thinning
  -lr , --low-ratio    Canny low threshold
  -hr , --high-ratio   Canny high threshold
  --width              Output width
  --height             Output height
  --fit                Fit printed result to terminal screen
  -o , --output        Filename for .txt file to save result
```



## Requirements

* [Matplotlib](https://matplotlib.org/) - image reading and basic plotting

* [NumPy](https://numpy.org/) – basic matrix (image) manipulation

* [SciPy](https://scipy.org/) – image convolution with kernels



## Future work

* ~~Change `-dx` and `-dy` parameters to `--height` and `--width`~~
* Support for non `.png` inputs
* Output as `.jpeg` instead of `.txt`
* Port to web application?
    * Drag + drop source image
    * Drag sliders to adjust parameters
    * WYSIWYG

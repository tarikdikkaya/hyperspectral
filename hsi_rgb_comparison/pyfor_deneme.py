from spectral import *
import numpy as np

img = open_image('BONA_017_2019_hyperspectral.hdr')

view = imshow(img, (57, 31, 17))
input("Press Enter to close...")

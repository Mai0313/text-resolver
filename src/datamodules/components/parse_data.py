import numpy as np
import os
import glob
from PIL import Image
from matplotlib import pyplot as plt

filename = 'image-test'

img = Image.open(filename + '.png')
data = np.array(img, dtype='uint8')

np.save(filename + '.npy', data)

# visually testing our output
img_array = np.load(filename + '.npy')
plt.imshow(img_array)


class PictureParser:

    def __init__(self, path):
        self.path = path

    def get_data_path(self):
        pass

    def get_parsed_data(self):
        pass

# -*- coding: utf-8 -*-


import sys

from labelme2voc_ import createMasks
# from image_utils import extension_change

input_dir = './image-registration-corrosion-growth/'
output_dir = './VOC_image-registration-corrosion-growth/'
label_txt_file = 'labels_corrosion_segmentation.txt'

createMasks(input_dir, output_dir, label_txt_file)

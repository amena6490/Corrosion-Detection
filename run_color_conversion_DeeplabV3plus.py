# -*- coding: utf-8 -*-

import numpy as np
from color_conversion_DeeplabV3plus import color_conversion_DeeplabV3plus

source = './image-registration-corrosion-growth/mask_512/'
destination = './image-registration-corrosion-growth/masks_bn/'
# source_mask_folder, destination, extension
color_conversion_DeeplabV3plus(source, destination, 'png')

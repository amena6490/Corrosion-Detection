# -*- coding: utf-8 -*-
import sys

sys.path.insert(0, 'S://Python/general_utils/')

from image_utils import extension_change

source_image_folder = './Test/Images/'
destination = './Test/Images/'
extension = 'jpg'
extension_change(source_image_folder, destination, extension)

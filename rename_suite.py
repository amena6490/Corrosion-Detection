# -*- coding: utf-8 -*-

from image_utils import rename_segmentation

source = './Test/Images/'
source_image_folder = './images/'
source_mask_folder = './VOC_440_Corrosion_Dataset_review/SegmentationClassPNG/'
source_json_folder = './JSON/'
source_ohev_folder = './VOC_440_Corrosion_Dataset_review/SegmentationClass/'

test_or_train = './Test_renamed/'

image_destination = '/images/'
mask_destination = '/masks/'
json_destination = '/json/'
ohev_destination = './ohev/'

rename_segmentation(source, source_image_folder, source_mask_folder, source_json_folder, source_ohev_folder,
                        test_or_train, image_destination, mask_destination, json_destination, ohev_destination)

""" DeepLabv3 Model download and change the head for your prediction"""
from network._deeplab import DeepLabHeadV3Plus # good
from network.modeling import deeplabv3plus_resnet50 # good
from network.modeling import deeplabv3plus_resnet101 # good
import torch


def createDeepLabv3Plus(outputchannels=4, output_stride=8):
    inplanes = 2048
    low_level_planes = 256
    num_classes = outputchannels
    model = deeplabv3plus_resnet101(num_classes=num_classes, output_stride=8)
    model.classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes)
    # Set the model in training mode
    model.train()
    return model

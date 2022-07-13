from torchvision import transforms
from collections import OrderedDict
import os
from PIL import Image
import torch

import cv2
import openvino.runtime as ov
import numpy as np
import time
from utils import Attention_AR_counter
import pandas as pd

img_width = 256
img_height = 64
transf = transforms.ToTensor()


def img_loader(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((img_width, img_height))
    img = transf(img)
    img = torch.unsqueeze(img,dim = 0)
    return img

def read_GT():
    df = pd.read_csv('./MAVI_clean_eng/gt_cropped_eng_1.txt', sep='\t')
    df.columns = ['path', 'word']
    return df['path'], df['word']

def load_model(path, device):
    core = ov.Core()
    model = core.read_model(path)
    compiled_model = core.compile_model(model, device)
    return compiled_model

def generateInferences(IMG_path):
    # load model
    path = './models/VisionLAN/VisionLAN.xml'
    device = 'CPU'
    compiled_model = load_model(path, device)
    # load image
    img = img_loader(IMG_path)
    request = compiled_model.create_infer_request()
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    request.infer(inputs={input_layer:img})
    result = request.get_output_tensor(output_layer.index).data
    out = torch.tensor(result)
    return out

def generateOutputText(prob):
    test_acc_counter = Attention_AR_counter('\ntest accuracy: ', './dict/dic_36.txt', False)
    pre_string = test_acc_counter.convert(prob)
    return pre_string[0]

def writeOutputFile():
    paths, word = read_GT()
    # list all images in the directory
    path_org = './MAVI_clean_eng/'
    paths = [path[2:] for path in paths]
    # append path_org to paths
    images = [path_org + path for path in paths]
    # create inference for all images in images
    for i in images:
        #img_path = path + i
        prob = generateInferences(i)
        text = generateOutputText(prob)
        #write text to file
        with open('./output_FP32.txt', 'a') as f:
            f.write(i + '\t' + text + '\n')

    print('output.txt written')


if __name__ == '__main__':
    writeOutputFile()
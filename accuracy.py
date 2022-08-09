from torchvision import transforms
from collections import OrderedDict
import os
from PIL import Image
import torch

import editdistance as ed

import cv2
import openvino.runtime as ov
import numpy as np
from utils import Attention_AR_counter
import pandas as pd

img_width = 256
img_height = 64
transf = transforms.ToTensor()

def TopK(x, k):
    a = dict([(i, j) for i, j in enumerate(x)])
    sorted_a = dict(sorted(a.items(), key = lambda kv:kv[1], reverse=True))
    indices = list(sorted_a.keys())[:k]
    values = list(sorted_a.values())[:k]
    return (indices, values)


def add_iter(prediction, label_length, labels):
    #print('Adding iter...')
    total_samples += label_length.size()[0]
    for i in range(0, len(prediction)):
        prediction[i] = prediction[i].lower()
        labels[i] = labels[i].lower()
        all_words = []
        for w in labels[i].split('|') + prediction[i].split('|'):
            if w not in all_words:
                all_words.append(w)
        l_words = [all_words.index(_) for _ in labels[i].split('|')]
        p_words = [all_words.index(_) for _ in prediction[i].split('|')]
        distance_C += ed.eval(labels[i], prediction[i])
        distance_W += ed.eval(l_words, p_words)
        total_C += len(labels[i])
        total_W += len(l_words)
        correct = correct + 1 if labels[i] == prediction[i] else correct
    return prediction, labels

def accuracy(correct, total,best_acc, change= False):
    #print('Calculating accuracy...')
    acc = correct/total  
    dist_c = 0
    total_c = 0 
    dist_w = 0
    total_w = 0 
    print('Accuracy: {:.6f}, AR: {:.6f}, CER: {:.6f}, WER: {:.6f}'.format(
        acc,
        1 - dist_c / total_c,
        dist_c / total_c,
        dist_w / total_w))
    return best_acc

def img_loader(path):
    #print('Loading image...')
    img = Image.open(path).convert('RGB')
    img = img.resize((img_width, img_height))
    img = transf(img)
    img = torch.unsqueeze(img,dim = 0)
    return img

def read_GT():
    #print('Reading GT...')
    #df = pd.read_csv('./mavi_hindi/gt_1.txt', sep='\t')
    df = pd.read_csv('./MAVI_clean_eng/gt_cropped_eng_1.txt', sep='\t')
    df.columns = ['path', 'word']
    return df['path'], df['word']

def load_model(path, device):
    #print('Loading model...')
    core = ov.Core()
    model = core.read_model(path)
    compiled_model = core.compile_model(model, device)
    return compiled_model

def generateInferences(IMG_path):
    #print('Generating inference...')
    # load model
    #path = './models/VL_MAVI/IR_HIN_FP32/VisionLAN_Hindi.xml'
    path = './models/VL_MAVI/IR_ENG_FP32/VisionLAN_ENG.xml'
    device = 'CPU'
    compiled_model = load_model(path, device)
    # load image
    img = img_loader(IMG_path)
    request = compiled_model.create_infer_request()
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    request.infer(inputs={input_layer:img})
    result = request.get_output_tensor(output_layer.index).data
    output = result
    #print(output.shape)
    i = 0
    for i in range(output.shape[0]):
        b = TopK(output[i], 1)[0]
        #print(b)
        if b == [0]:
            break
    #print(i)
    output = output[0:i]
    #out = torch.tensor(result)
    output = torch.tensor(output)
    return output

def generateOutputText(prob):
    #print('Generating output text...')
    #test_acc_counter = Attention_AR_counter('\ntest accuracy: ', './dict/dict_hindi.txt', False)
    test_acc_counter = Attention_AR_counter('\ntest accuracy: ', './dict/dic_36.txt', False)
    pre_string = test_acc_counter.convert(prob)
    return pre_string[0]

def writeOutputFile():
    #print('Writing output file...')
    paths, word = read_GT()
    #print(paths)
    # list all images in the directory
    #path_org = './mavi_hindi/four/1/'

    paths = [path[:] for path in paths]
    #print(paths)
    #exit()
    # append path_org to paths
    #images = [path_org + path for path in paths]
    images = ['./MAVI_clean_eng/' + path[2:] for path in paths]
    #print(images)
    #exit()
    # create inference for all images in images
    for i in images:
        #img_path = path + i
        prob = generateInferences(i)
        text = generateOutputText(prob)
        print(text)
        #write text to file
        with open('./output_FP32_MAVI_ENG.txt', 'a') as f:
            f.write(i + '\t' + text + '\n')

    print('output.txt written')


if __name__ == '__main__':
    writeOutputFile()
    #print('Hi')
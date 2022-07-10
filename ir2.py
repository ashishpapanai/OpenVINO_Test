from torchvision import transforms
from collections import OrderedDict
import os
from PIL import Image
import torch

import cv2
import openvino.runtime as ov
import numpy as np
import time

core=ov.Core()

st_time=time.time()
model=core.read_model("./models/VisionLAN/VisionLAN.xml")
compiled_model=core.compile_model(model,"MYRIAD")
ed_time=time.time()
print("loading_time: ",ed_time-st_time)


img_width = 256
img_height = 64

transf = transforms.ToTensor()


img = Image.open('./img.jpeg').convert('RGB')
img = img.resize((img_width, img_height))
img = transf(img)
img = torch.unsqueeze(img,dim = 0)
target = ''


request=compiled_model.create_infer_request()
input_layer=compiled_model.input(0)
output_layer=compiled_model.output(0)
ie_time=time.time()
request.infer(inputs={input_layer:img})
result=request.get_output_tensor(output_layer.index).data
ien_time=time.time()
print("inference_time: ",ien_time-ie_time)
print(result)
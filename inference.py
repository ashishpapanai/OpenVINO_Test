import cv2
import openvino.runtime as ov
import numpy as np
import time

core=ov.Core()

st_time=time.time()
model=core.read_model("./models/SVTR/SVTR_Tiny.xml")
compiled_model=core.compile_model(model,"MYRIAD")
ed_time=time.time()
print("loading_time: ",ed_time-st_time)


def resize_svtr(img):
    img=cv2.resize(img,(256,64),interpolation=cv2.INTER_LINEAR)
    resized_img=img.astype(np.float32)
    resized_img=resized_img.transpose((2,0,1))/255.0
    resized_img -= 0.5
    resized_img /= 0.5
    return resized_img

img=cv2.imread("./img.jpeg")
norm_img=resize_svtr(img)
norm_img=norm_img[np.newaxis,:]


request=compiled_model.create_infer_request()
input_layer=compiled_model.input(0)
output_layer=compiled_model.output(0)
ie_time=time.time()
request.infer(inputs={input_layer:norm_img})
result=request.get_output_tensor(output_layer.index).data
ien_time=time.time()
print("inference_time: ",ien_time-ie_time)
print(result)
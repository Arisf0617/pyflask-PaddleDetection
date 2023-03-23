# 推理部署
import requests
from PIL import Image
from io import BytesIO

import cv2
import numpy as np
from PaddleDetection.deploy.python.infer import Detector, visualize_box_mask

model_dir = "模型路径"

detector = Detector(
    model_dir,
    device='CPU',
    run_mode='paddle',
    trt_min_shape=1,
    trt_max_shape=1280,
    trt_opt_shape=640,
    trt_calib_mode=False,
    cpu_threads=1,
    enable_mkldnn=False,
    enable_mkldnn_bfloat16=False,
    output_dir='all_result',
    threshold=0.1)

print("***********MODEL LOADED!***********")
img_path = r'图片地址http'

#读取网络图片
# cv2无法直接读取网络图片，将PIL.Image转换成OpenCV格式
response = requests.get(img_path)
response = response.content
BytesIOObj = BytesIO()
BytesIOObj.write(response)
img = Image.open(BytesIOObj)
# ********结束********
frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
results = detector.predict_image([frame[:, :, ::-1]], visual=False)  # 保存图片有问题，不开启
# ********处理list返回值********
rs = visualize_box_mask(frame, results, detector.pred_config.labels, detector.threshold)
im = rs[0]
label_json = rs[1]
# ********处理list返回值结束********
print(label_json) # 输出json格式的标签
im = np.array(im)
cv2.imshow('Detection', im) # 展示图片
cv2.imwrite('all_result/111.jpg', im) # cv2保存图片
cv2.waitKey(0)


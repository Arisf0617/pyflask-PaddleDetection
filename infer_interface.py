# 推理Flask接口化
import os
import requests
from PIL import Image
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PaddleDetection.deploy.python.infer import Detector, visualize_box_mask

model_dir = r""  # 模型路径
save_path = r""  # 推理结果保存路径

# 推理参数设置
detector = Detector(
    model_dir,
    device='GPU',
    run_mode='paddle',
    trt_min_shape=1,
    trt_max_shape=1280,
    trt_opt_shape=640,
    trt_calib_mode=False,
    cpu_threads=1,
    enable_mkldnn=False,
    enable_mkldnn_bfloat16=False,
    output_dir=save_path,
    threshold=0.1)

def infer_start(img_path, img_name):
    # 读取网络图片
    # cv2无法直接读取网络图片，将PIL.Image转换成OpenCV格式
    response = requests.get(img_path)
    response = response.content
    BytesIOObj = BytesIO()
    BytesIOObj.write(response)
    img = Image.open(BytesIOObj)
    # ********结束********
    frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    print("***********INFER START!***********")
    results = detector.predict_image([frame[:, :, ::-1]], visual=False)  # 保存图片有问题，不开启
    # ********处理list返回值********
    rs = visualize_box_mask(frame, results, detector.pred_config.labels, detector.threshold)
    im = rs[0]
    label_rs = rs[1]
    # ********处理list返回值结束********
    im = np.array(im)
    cv2.imwrite(os.path.join(save_path) + img_name.replace(':', '_'), im)  # cv2保存图片
    return label_rs

app = Flask(__name__)
CORS(app, resources=r'/*')

@app.route('/infer', methods=['POST'])
def main():
    image_url = request.args.get("image")  # 获得图片路径URL
    img_name = os.path.split(image_url)[1]  # 获取图片名称
    # os.path.join在Windows下拼接会显示'\\'而非'/'，使用replace替换路径中的'\\'
    save_rs = (os.path.join(os.getcwd(), save_path, img_name)).replace('\\', '/').replace(':', '_')
    print("接收到的图片地址：" + image_url)
    result = infer_start(image_url, img_name)
    # return result
    print("***********INFER END!***********")
    return jsonify({'code': 200, 'save_path': save_rs, 'message': result})


app.run(port=8000) # 接口的运行端口

import os
import cv2
import requests
from PIL import Image
from io import BytesIO
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

# 加载paddlex，设置GPU使用环境，如果非GPU环境会自动使用CPU
import paddlex as pdx
os.environ['CUDA_VISIBLE_DEVICE'] = '0'

#test_path = 'images/'
save_path = 'all_result/'

# 模型加载
model_dir = 'best_model2'
model = pdx.load_model(model_dir)

font = cv2.FONT_HERSHEY_SIMPLEX
def Picture_frame(img,box_list):
    for item in box_list:
        # 接受阈值
        if(item['score']>0.4):
            x = int(item['bbox'][0])
            y = int(item['bbox'][1])
            w = x +int(item['bbox'][2])
            h = y +int(item['bbox'][3])
            cv2.rectangle(img, (x,y), (w, h), (0, 255, 0), 2)
            text = item['category']+str(round(item['score'],3))
            print(item['category'],item['score'])
            cv2.putText(img, text, (x, y), font, 0.9, (0, 0, 255), 2)
        # else:
        #     return img
    return img

def main(img_path):
    print("图片路径："+img_path)
    img_name = os.path.split(img_path)[1]

    #cv2无法直接读取网络图片，将PIL.Image转换成OpenCV格式
    response = requests.get(img_path)
    response = response.content
    BytesIOObj = BytesIO()
    BytesIOObj.write(response)
    img = Image.open(BytesIOObj)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    #img = cv2.imread(img_path)
    #print(img)

    result = model.predict(img)
    img = Picture_frame(img, result)

    cv2.imwrite(os.path.join(save_path) + img_name, img)#保存

    return os.path.join(save_path)

from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app,resources=r'/*')
@app.route('/detect', methods=['POST'])
def login():
    image = request.args.get("image")
    print(image)
    out_path = main(image)
    return jsonify({'code':200,'message':out_path})

app.run(port=8000)

# infer_img = cv2.imread(os.path.join('all_result/','2016.jpeg'))
# plt.figure(figsize=(100,100))
# plt.imshow(cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB))
# plt.show()
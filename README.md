# pyflask-PaddleDetection
根据PaddleDetection源码自定义部署的python程序，Flask接口版和本地版

使用PIL.Image将网络图片地址转化为cv2格式，解决了cv2无法直接读取网络图片的问题

！！！模型需要先进行推理导出后才可部署，导出流程见官方文档

1.clone PaddleDetection源码至本地项目

PaddleDetection项目地址：https://github.com/PaddlePaddle/PaddleDetection

git clone https://github.com/PaddlePaddle/PaddleDetection.git

2.import推理类

from PaddleDetection.deploy.python.infer import Detector, visualize_box_mask

注：接口版中代码的58-60行和本地版40-42行，为本人根据项目需求修改visualize_box_mask()源码后的返回结果，因此使用前，需根据自身需要更改返回值

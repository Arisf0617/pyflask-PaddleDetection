# pyflask-PaddleDetection
根据PaddleDetection源码自定义部署的python程序，Flask接口版和本地版

1.clone PaddleDetection源码至本地项目
PaddleDetection项目地址：https://github.com/PaddlePaddle/PaddleDetection

git clone https://github.com/PaddlePaddle/PaddleDetection.git

2.import推理类

from PaddleDetection.deploy.python.infer import Detector, visualize_box_mask

注：Flask接口版中代码的58-60行为本人根据项目需要修改visualize_box_mask()源码后的结果，因此使用前需根据自身需要更改返回值

FROM ultralytics/ultralytics:latest-jetson-jetpack4

RUN pip3 uninstall onnxruntime-gpu -y

RUN wget https://nvidia.box.com/shared/static/2sv2fv1wseihaw8ym0d4srz41dzljwxh.whl -O onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl

RUN pip3 install onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl

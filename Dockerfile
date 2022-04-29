FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get -y update && apt-get -y install git build-essential
RUN pip install cython scipy docopt opencv-python-headless scikit-image
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'


# Quick Start(for Jetson)

## Reference
Our jetson environment follows [jkjung](https://github.com/jkjung-avt)'s [jetson-nano](https://github.com/jkjung-avt/jetson_nano)


## Step 1

To install the TensorFlow on Nvidia Jetson devices, please refer to [here](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html). Protobuf 3.8.0 also has to be installed as follows:

```shell
$ mkdir ${HOME}/project
$ cd ${HOME}/project
$ git clone https://github.com/jkjung-avt/jetson_nano.git
$ cd jetson_nano
$ ./install_protobuf-3.8.0.sh
$ source ${HOME}/.bashrc
```

## Step 2

To install dependencies and build the TensorRT Engine, please enter the commands as follow.

```Shell
$ cd ${HOME}/project
$ git clone https://github.com/jkjung-avt/tensorrt_demos.git
$ cd ${HOME}/project/tensorrt_demos/ssd
$ ./install_pycuda.sh
$ sudo pip3 install onnx==1.4.1
$ cd ${HOME}/project/tensorrt_demos/plugins
$ make
```


## Step 3

To use this runtime module, you need to clone this repository in your local storage and install dependencies that are described in ‘requirements.txt’.

```shell
$ git clone https://github.com/nota-github/modelsearch-runtime.git
$ pip3 install -r requirements.txt
```


To execute the ‘runtime.py’, you can use an example model with sample images that we provide, which can be accessed on this link below.

- https://drive.google.com/drive/u/1/folders/1BPzFRem_pu3qpTuTvVAJ8gouwq1yRERN



Example command:

```shell
# TFLite
$ python3 runtime.py --model ./runtime_example/example.tflite --image_folder ./runtime_example/ --classes ./runtime_example/data.yaml
# TensorRT
$ python3 runtime.py --model ./runtime_example/example.trt --image_folder ./runtime_example/ --classes ./runtime_example/data.yaml
```

- --model : a path where .trt or .tflite model file is located
- --image_folder : a path where images are located
- --classes : a path where a ‘yaml file’ that contains classes information are located


# NetsPresso-Model-Searcher-Runtime

This repository involves the source codes for inferring models in the form of the TensorRT(.trt) and Tensorflow Lite(.tflie) and Openvino. The visualization for the inference results is also supported.


## How to run
* mode: "save" for saving inference result image and "show" for showing inference image.
* conf_thres: Confidence threshold, default=0.25.
* iou_thres: IoU threshold, default=0.60.
* classes: Yaml file path of classes information.
```bash
$ python3 runtime.py --model ${model_file_path} --image_folder ${img_folder} --classes ${yaml_file_path} --mode save
```

Please refer to [getting_started.md](https://github.com/Nota-NetsPresso/NetsPresso-ModelSearch-Runtime/blob/main/getting_started.md) and [example](https://github.com/Nota-NetsPresso/NetsPresso-ModelSearch-Runtime/blob/main/examples/readme.md) for more details.

## Result

![voc_0 (1)](https://user-images.githubusercontent.com/69896052/138831707-bba0f0f8-c74a-47cc-b44a-c2a1a44ff898.jpg)
![voc_1 (1)](https://user-images.githubusercontent.com/69896052/138831719-b099c21a-5feb-43a4-be33-c190323e3f0a.jpg)
![coco2 (1)](https://user-images.githubusercontent.com/69896052/138831835-4cc4e0c3-62e0-4248-9d8b-ea155f348019.jpg)

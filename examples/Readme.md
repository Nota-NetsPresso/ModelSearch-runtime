# NetsPresso whl package

To create NetsPresso whl package, please modify the 'example.py' to fit the model file you want to use.

## A common part
1. preprocess and postprocess functions must be defined and can be modified if necessary. As defined in the example file, the output of the preprocess must be in a form that the model can infer. The input of postprocess must be the result of the model's inference.


2. Write model class name.
```python
class {model_class_name}(Basemodel):
```

3. Write classes name in the same order of datasets.

![classes](https://user-images.githubusercontent.com/45225793/185578403-65c3917f-8e76-4cd3-b72e-3f235ff244cb.png)

```python
    self.classes = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", 
        "diningtable", "dog", "horse", "motorbike", "person", 
        "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    ]

```


## For Tensorflow Lite
Write "self.input_layer_location" and "self.output_layer_location" the same as the model. "self.input_layer_name" and "self.output_layer_name" are empty list.

```python
  self.input_layer_name = []
  self.input_layer_location = [0]
  self.output_layer_name = []
  self.output_layer_location = [549]
```
### Get location numbers of input and output of Tensorflow Lite model
Open Tensorflow Lite model with [Netron](https://netron.app/)
![tflite input](https://user-images.githubusercontent.com/45225793/185898563-dc8a9a74-2a6d-49ba-96ee-a6d15dc0b0cc.png)
![tflite output](https://user-images.githubusercontent.com/45225793/185898575-9ea4014a-cc74-4f45-847c-1a2b03dcf821.png)


## For TensorRT and OpenVINO
Write "self.input_layer_name" and "self.output_layer_name" the same as the model. "self.input_layer_location" and "self.output_layer_location" are empty lists.

```python
  self.input_layer_name = ["images"]
  self.input_layer_location = []
  self.output_layer_name = ["output"]
  self.output_layer_location = []
```
### Get layer names of input and output of TensorRT
Load and print layer names.
```python
try:
    import tensorrt
except ImportError:
    raise ImportError("Failed to load tensorrt")

import os
import pdb
from pathlib import Path

def get_layer_names(model_path):
    TRT_LOGGER = tensorrt.Logger(tensorrt.Logger.WARNING)
    runtime = tensorrt.Runtime(TRT_LOGGER)
    input_names = []
    output_names = []
    with open(model_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        for binding in engine:
            if engine.binding_is_input(binding):
                input_names.append(binding)
            else:
                output_names.append(binding)

    print("Input layer names: ", input_names)
    print("Output layer names: ", output_names)

get_layer_names("/app/model.engine")
```
```bash
$ python run.py 
Input layer names:  ['unique_ids_raw_output___9:0', 'segment_ids:0', 'input_mask:0', 'input_ids:0']
Output layer names:  ['unique_ids:0', 'unstack:1', 'unstack:0'] 
```
### Get layer names of input and output of OpenVINO
Open '.xml' file of model, find layer name.
![OpenVINO input](https://user-images.githubusercontent.com/45225793/185899327-d61471ae-c39a-45a8-930b-4bf256fd9071.png)
![OpenVINO output](https://user-images.githubusercontent.com/45225793/185899334-357d31e1-b9e6-4189-9b59-b7fca384c78f.png)


## Run
* package_name: Package name which user choose when create package.
* model_class_name: Model class name which user choose when create package.
* num_thread: Supported on Tensorflow Lite only. In case of TensorRT and OpenVINO, initialize without any parameter.
```python
from np_{package_name}.models.model import NPModel # model_class_name


NPModel.initialize(num_threads=1) # Initialize, 'num_thread' is supported on Tensorflow Lite only
npmodel = NPModel()
image_path = "/test_image.jpg" # Read image file
print(npmodel.run(image_path)) # Inference
NPModel.finalize() # Memory management
```

### For more information, contact us netspresso@nota.ai


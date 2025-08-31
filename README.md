# objectscope
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/agbleze/objectscope/.github%2Fworkflows%2Fci-cd.yml)
![GitHub Tag](https://img.shields.io/github/v/tag/agbleze/objectscope)
![GitHub Release](https://img.shields.io/github/v/release/agbleze/objectscope)
![GitHub License](https://img.shields.io/github/license/agbleze/objectscope)

ObjectScope extends Detectron2 with utilities and abstractions that simplifies and customizes model training and evaluation to the needs of your data and most dominant workflows. Several intermediate steps such as registering dataset, finetune anchor boxes, model optimization and quantization are provided out-of-box with a single command.


## Installation

To install and run objectscope successfully, you need to have Detectron2 installed. Incase you are using a gpu enabled device, then install objectscope with pytorch cuda version.

##### Install Detectron2 

```bash
pip install git+https://github.com/facebookresearch/detectron2.git
```

##### Install objectscope with cuda version of pytorch

```bash
pip install objectscope --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu118

```

## Usage

Objectscope can run from both terminal / commandline and python file or jupyter notebook.

### Train model in python file / Jupyternote book

To train a model, provide the arguments to initialize TrainSession class and call run method to start the model training. This is highlighted as follows:

```python
from objectscope.trainer import TrainSession

trainer = TrainSession(train_img_dir="TRAIN_IMG_DIR",
                            train_coco_json_file="TRAIN_COCO_JSON_FILE",
                            test_img_dir="TEST_IMG_DIR",
                            test_coco_json_file="TEST_COCO_JSON_FILE",
                            config_file_url="CONFIG_FILE_URL",
                            num_classes="NUM_CLASSES",
                            train_data_name="train_data_name",
                            test_data_name="test_data_name",
                            train_metadata={},
                            test_metadata={},
                            output_dir="OUTPUT_DIR",
                            device="cuda,
                            num_workers=4,
                            imgs_per_batch=8,
                            base_lr=0.0001,
                            max_iter=5,
                            checkpoint_period=1,
                        )
    trainer.run()
```

### Train model using terminal command

With a single command from the terminal, you can do alot more in addition to training the model including model optimization and export to onnx, model evaluation, visualizing model hyperparameters on tensorboard among others.

objectscope accepts both declaring the parameters to pass to the terminal command in an .env file, environment variable and passing it as commandline argument. When both are present for the same argument, parameters passed using the commandline are prioritized over others.

Example of training model from the terminal is as follows:

```bash
objectscope --train_img_dir "train"\
            --test_img_dir "test" \
            --output_dir "output_dir" \
            --train_coco_json_file "train_annotations.coco.json" \
            --test_coco_json_file "test_annotations.coco.json" \
            --max_iter 5 --num_classes 15 --checkpoint_period 1  --roi_heads_score_threshold 0.5 \
            --imgs_per_batch 4 --num_workers 10 --config_file_url "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml" \
            --device "cuda" --base_lr 0.00005 \
            --show_tensorboard --optimize_model
```

Example of declaring variables in the .env file to be accessible to objectscope is depicted as follows:

```.env
TRAIN_IMG_DIR="train_images"\
test_img_dir="test_images" \
OUTPUT_DIR="output_dir" \
TRAIN_COCO_JSON_FILE="train_annotations.coco.json" \
TEST_COCO_JSON_FILE="test_annotations.coco.json" \
MAX_ITER=100000 
NUM_CLASSES=15 
CHECKPOINT_PERIOD=100  
ROI_HEADS_SCORE_THRESHOLD=0.5
IMGS_PER_BATCH=4 
NUM_WORKERS=10 
CONFIG_FILE_URL="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
DEVICE="cuda" 
BASE_LR=0.00005
SHOW_TENSORBOARD=true
OPTIMIZE_MODEL=true
```


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`objectscope` was created by Agbleze. It is licensed under the terms of the MIT license.

## Credits

`objectscope` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

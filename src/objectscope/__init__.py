# read version from installed package
from importlib.metadata import version
import logging
from decouple import config
from typing import Union

package_name = __name__
__version__ = version(package_name)

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    filename="train_session.logs",
                    filemode="a"
                    )

logger.info(f"{package_name} version {__version__} initialized.")


TRAIN_IMG_DIR = config("TRAIN_IMG_DIR", default=None, cast=str)
test_img_dir = config("TEST_IMG_DIR", default=None, cast=str)
train_coco_json_file = config("TRAIN_COCO_JSON_FILE", default=None, cast=str)
test_coco_json_file = config("TEST_COCO_JSON_FILE", default=None, cast=str)
output_dir = config("OUTPUT_DIR", default="output/object_detector", cast=str)
num_classes = config("NUM_CLASSES", default=None)
num_workers = config("NUM_WORKERS", default=None)
imgs_per_batch = config("IMGS_PER_BATCH", default=4, cast=int)
base_lr = config("BASE_LR", default=0.00005, cast   =float)
max_iter = config("MAX_ITER", default=5000, cast=int)
checkpoint_period = config("CHECKPOINT_PERIOD", default=50, cast=int)
device = config("DEVICE", default="cuda", cast=str)
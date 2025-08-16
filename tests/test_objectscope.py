from cpauger.generate_coco_ann import generate_random_images_and_annotation
from objectscope.trainer import TrainSession
import os
from objectscope.evaluator import Evaluator
from objectscope import logger
from detectron2.engine import DefaultTrainer
import pytest
import tempfile
import pandas as pd

tempdir = tempfile.TemporaryDirectory()
train_img_dir = os.path.join(tempdir.name,"train_random_images")
train_coco_json_file=os.path.join(tempdir.name,"train_generated_annotation.json")
test_img_dir=os.path.join(tempdir.name,"test_random_images")
test_coco_json_file=os.path.join(tempdir.name, "test_generated_annotation.json")
train_data_name="random_train"
test_data_name="random_test"
output_dir=os.path.join(tempdir.name, "random_model_train")


train_imgpaths, train_coco_path = generate_random_images_and_annotation(image_height=224, image_width=224,
                                                                        number_of_images=10, 
                                                                        output_dir=train_img_dir,
                                                                        img_ext ="jpg",
                                                                        image_name="train_random_images",
                                                                        parallelize=True,
                                                                        save_ann_as=train_coco_json_file,
                                                                        )


test_imgpaths, test_coco_path = generate_random_images_and_annotation(image_height=224, image_width=224,
                                                                    number_of_images=5, 
                                                                    output_dir=test_img_dir,
                                                                    img_ext ="jpg",
                                                                    image_name="test_random_images",
                                                                    parallelize=True,
                                                                    save_ann_as=test_coco_json_file,
                                                                    )

@pytest.fixture(scope="class")
def create_train_session():
    trainer = TrainSession(train_img_dir=train_img_dir,
                        train_coco_json_file=train_coco_json_file,
                        test_img_dir=test_img_dir,
                        test_coco_json_file=test_coco_json_file,
                        config_file_url="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                        num_classes=3,
                        train_data_name=train_data_name,
                        test_data_name=test_data_name,
                        train_metadata={},
                        test_metadata={},
                        output_dir=output_dir,
                        device="cpu", num_workers=4,
                        imgs_per_batch=8,
                        base_lr=0.0001,
                        max_iter=3,
                        checkpoint_period=1,
                        )
    return trainer

@pytest.fixture(scope="class")
def get_config(create_train_session):
    cfg, _ = create_train_session.create_config()
    return cfg

@pytest.fixture(scope="class")
def create_evaluator(create_train_session, get_config
                     ):
    trainer = create_train_session
    cfg = get_config
    evaluator = Evaluator(cfg=cfg,
                        test_data_name=trainer.test_data_name,
                        output_dir=output_dir,
                        dataset_nm=trainer.test_data_name,
                        metadata=trainer.test_metadata,
                        roi_heads_score_threshold=0.5                            
                        )
    return evaluator

@pytest.fixture
def create_evaluation_result(create_evaluator, get_config):
    evaluator = create_evaluator
    eval_df = evaluator.evaluate_models(cfg=get_config)
    return eval_df


def test_trainer(create_train_session):
    trainer_obj = create_train_session.run()
    assert isinstance(trainer_obj, DefaultTrainer)

def test_evaluate_models(create_evaluation_result):
    assert isinstance(create_evaluation_result, pd.DataFrame)

def test_get_best_model(create_evaluation_result, create_evaluator):
    eval_df = create_evaluation_result
    evaluator = create_evaluator
    best_model_res = evaluator.get_best_model(eval_df)
    assert isinstance(best_model_res, dict)
    assert "best_model_name" in best_model_res.keys()
        

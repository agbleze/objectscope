import torch
import pickle
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.export import TracingAdapter
from typing import Union
from objectscope import logger
from detectron2.export import scripting_with_instances
from torch import Tensor
from detectron2.structures import Boxes

fields = {"proposal_boxes": Boxes,
    "objectness_logits": Tensor,
    "pred_boxes": Boxes,
    "scores": Tensor,
    "pred_classes": Tensor,
    "pred_masks": Tensor,
    "pred_keypoints": Tensor,
    "pred_keypoint_heatmaps": Tensor,
    }

class OnnxModelExporter(object):
    def __init__(self, cfg_path, model_path, 
                 registered_dataset_name
                 ):
        self.model_path = model_path
        self.registered_dataset_name = registered_dataset_name
        with open(cfg_path, "rb") as f:
            self.cfg = pickle.load(f)
        self.cfg.MODEL.WEIGHTS = self.model_path
    
    def get_traceadapted_model(self, 
                               model: Union[DefaultPredictor, None]=None,
                               inputs=None
                               ):
        if not model:
            if not hasattr(self, "model"):
                model = self.get_predictor()
            else:
                model = self.model
        if not inputs:
            if hasattr(self, "inputs"):
                inputs = self.inputs
            else:
                inputs = self.get_sample_model_inputs()
        self.wrapper = TracingAdapter(model, inputs=inputs)
        return self.wrapper
    
    def get_predictor(self, cfg=None):
        if not cfg:
            cfg = self.cfg
        predictor = DefaultPredictor(cfg)
        self.model = predictor.model
        return self.model
    
    def get_sample_model_inputs(self, cfg=None, 
                                registered_dataset_name=None
                                ):
        if not cfg:
            cfg = self.cfg
        if not registered_dataset_name:
            registered_dataset_name = self.registered_dataset_name
        dataloader = build_detection_test_loader(cfg, registered_dataset_name)
        loaded_data = iter(dataloader)
        self.inputs = next(loaded_data)
        self.inputs = [{"image": input["image"] for input in self.inputs}]
        return self.inputs
    
    def export_to_onnx(self, save_onnx_as, inputs=None,
                        model=None
                        ):
        traced_model = self.get_traceadapted_model(model=model,
                                                   inputs=inputs
                                                   )
        inputs = inputs if inputs else self.inputs
            
        with open(save_onnx_as, "wb") as f:
            image = inputs[0]["image"]
            torch.onnx.export(model=traced_model.eval(),
                            args = (image,),
                            f = f, opset_version=16
                            )
        logger.info(f"Successfully exported model to onnx at: {save_onnx_as}")
        
    def create_script_model(self, 
                            model: Union[DefaultPredictor, None]=None,
                            fields=fields
                            ):
        if not model:
            if hasattr(self, "model"):
                model = self.model
            if not hasattr(self, "model"):
                model = self.get_predictor()
        self.scripted_model = scripting_with_instances(model.eval(), 
                                                  fields=fields
                                                  )
        return self.scripted_model

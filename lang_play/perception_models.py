import os
import json
import random
import numpy as np
import torch
import torchvision

# TODO : Install detectron2
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from models import PerceptionModule

config_file_map =  {'instance_segmentation' : 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
                    'panoptic_segmentation' : 'COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml',
                    'object_detection': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'}

func_map = {'instance_segmentation' : instance_segmentation,
            'panoptic_segmentation' : panoptic_segmentation}

def set_up_detectron(config_file):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

def panoptic_segmentation(cfg, model, image):
    outputs = model(image)
    # Might have to use this : draw_panoptic_seg_predictions
    return outputs["panoptic_seg"][0] 

def instance_segmentation(cfg, model, image):
    raise NotImplementedError

class ObjectMask(torch.nn.Module):
    '''
        Maps raw images into their segment (and depth) masks.
        A mostly-pretrained model (other than perception_module) to provide a better representation.
    '''
    def __init__(self, segment_type='panoptic', depth=False, 
        visual_obv_dim=[200, 200, 3], dof_obv_dim=[8], state_dim=64):
        super(ObjectMask, self).__init__()
        
        self.depth = depth
        
        assert segment_type in ['instance', 'panoptic']
        self.segment_type = segment_type.lower() + "_segmentation"
        self.segmentation_model, self.cfg = set_up_detectron(config_file_map[self.segment_type])
        self.segmentation_callback = func_map[self.segment_type]

        self.perception_module = PerceptionModule(visual_obv_dim, dof_obv_dim, state_dim)

    def forward(self, visual_obv, dof_obv=None):
        visual_obv = self.segmentation_callback(self.cfg, self.segmentation_model, visual_obv)
        output = self.perception_module(visual_obv, dof_obv)
        return output

class ObjectStateFromInstruction(torch.nn.Module):
    ''' 
        Gets the object state (position, keypoints) for the object specified in the instruction - based on it's label.
    '''
    def __init__(self):
        super(ObjectStateFromInstruction, self).__init__()
        self.detection_model, self.cfg = set_up_detectron(config_file_map['object_detection'])

    def detect_objects(self, image):
        outputs = self.detection_model(image)
        classes, boxes = outputs["instances"].pred_classes, outputs["instances"].pred_boxes
        keypoints = outputs["instances"].pred_keypoints
        
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        label_dict = metadata.get("thing_classes", None)
        labels = [label_dict[i] for i in classes]

        outputs = [[classes[i], labels[i], boxes[i]] for i in range(len(lables))] 
        # TODO : Include keypoints maybe
        # TODO : If not, just return boxes
        return outputs, labels

    def object_idx_in_instruction(self, labels, instruction):
        # TODO : *IMPORTANT* change this - PLEASE.
        for i, label in enumerate(labels):
            if label in instruction:
                return i

    def forward(self, image, instruction):
        # TODO - Concatenate w/ dof obvs and return- Currently returns the corresponding object representation 
        objects, labels = detect_objects(image)
        object_idx = object_idx_in_instruction(labels, instruction)

        return objects[object_idx][2]
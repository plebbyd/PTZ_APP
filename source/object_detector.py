import os
import json
import re
import torch
import numpy as np
from typing import Union, List, Optional, Tuple, Dict
from abc import ABC, abstractmethod
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM
from pathlib import Path

class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, image: Image.Image, target_objects: Union[str, List[str]]) -> Tuple[List[float], List[List[int]], List[str]]:
        pass
    @abstractmethod
    def load_model(self):
        pass

class YOLODetector(ObjectDetector):
    def __init__(self, model_name: str, **kwargs): # Accept kwargs to ignore context
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        try:
            print(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(f"{self.model_name}.pt")
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model {self.model_name}. Error: {str(e)}")

    def detect(self, image: Image.Image, target_objects: Union[str, List[str]]) -> Tuple[List[float], List[List[int]], List[str], dict]:
        if isinstance(target_objects, str):
            target_objects = [target_objects]
        target_objects = [obj.lower() for obj in target_objects]
        image_np = np.array(image)
        results = self.model(image_np)
        bboxes, labels, rewards = [], [], []
        for r in results:
            for box in r.boxes:
                cls = r.names[int(box.cls[0])]
                if "*" in target_objects or cls.lower() in target_objects:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bboxes.append([int(x1), int(y1), int(x2), int(y2)])
                    labels.append(cls)
                    rewards.append(1 - float(box.conf[0]))
        return rewards, bboxes, labels, {}

class FlorenceDetector(ObjectDetector):
    def __init__(self, model_name: str, prompt_context: Optional[str] = None):
        self.model_size = 'base' if 'base' in model_name.lower() else 'large'
        self.prompt_context = prompt_context or "In this scene"
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.load_model()

    def load_model(self):
        model_path = f"/hf_cache/microsoft/Florence-2-{self.model_size}"
        print(f"Loading Florence model directly from path: {model_path}")
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        print(f"Successfully loaded Florence model from {model_path}.")

    def detect(self, image: Image.Image, target_objects: Union[str, List[str]]) -> Tuple[List[float], List[List[int]], List[str], dict]:
        if target_objects == "*" or (isinstance(target_objects, list) and "*" in target_objects):
            task = "<OD>"
            text = f"<OD> {self.prompt_context}"
        else:
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            joined_objects = ", ".join(target_objects)
            text = f"{task} {self.prompt_context}, find any of the following: {joined_objects}. Ignore any irrelevant man-made objects."
        
        print(f"Using prompt: {text}")
        
        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        results = self.processor.post_process_generation(generated_text, task=task, image_size=image.size)
        
        print(f"Raw results: {results}")
        task_results = results.get(task, {})
        bboxes = task_results.get('bboxes', [])
        labels = task_results.get('labels', [])
        rewards = [0.5] * len(bboxes)
        return rewards, bboxes, labels, results

class DetectorFactory:
    @staticmethod
    def create_detector(model_name: str, target_objects: Union[str, List[str]], prompt_context: Optional[str] = None) -> 'ObjectDetector':
        model_name = model_name.lower()
        if 'florence' in model_name:
            return FlorenceDetector(model_name, prompt_context)
        elif 'yolo' in model_name:
            return YOLODetector(model_name)
        else:
            raise ValueError("Invalid model type specified.")

def get_label_from_image_and_object(
    image: Image.Image,
    target_object: str,
    detector: ObjectDetector,
    processor=None
) -> List[Dict]:
    rewards, bboxes, labels, raw_results = detector.detect(image, target_object)
    results = []
    for reward, bbox, label in zip(rewards, bboxes, labels):
        results.append({'reward': reward, 'bbox': bbox, 'label': label})
    return (results, raw_results) if results else ([], {})

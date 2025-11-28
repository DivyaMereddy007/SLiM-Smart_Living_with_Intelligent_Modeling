import torch
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq, 
    CLIPProcessor, 
    CLIPModel,
    pipeline
)
from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from PIL import Image
import supervision as sv
import cv2
from typing import List, Dict, Tuple, Any
import os

class EnhancedModelLoader:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kosmos = None
        self.kosmos_processor = None
        self.clip_model = None
        self.clip_processor = None
        self.grounding_dino = None
        self.sam_predictor = None
        
    def load_models(self):
        """Load all models required for enhanced detection and labeling, with error logging."""
        try:
            print("Loading BLIP vision-language model (Salesforce/blip-image-captioning-base)...")
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.kosmos_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.kosmos = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            print("BLIP model loaded and moved to device.")
        except Exception as e:
            print(f"Error loading BLIP: {e}")
            raise

        try:
            print("Loading CLIP...")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            raise

        try:
            print("Loading GroundingDINO...")
            config_path = os.path.join(os.path.dirname(__file__), "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
            weights_path = os.path.join(os.path.dirname(__file__), "groundingdino", "weights", "groundingdino_swint_ogc.pth")
            self.grounding_dino = load_model(
                config_path,
                weights_path
            )
        except Exception as e:
            print(f"Error loading GroundingDINO: {e}")
            raise

        try:
            print("Loading SAM...")
            sam_checkpoint = "sam_vit_h_4b8939.pth"
            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
        except Exception as e:
            print(f"Error loading SAM: {e}")
            raise
        
    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process an image through all models to get comprehensive analysis.
        Returns detailed object detection and classification results.
        """
        # Convert PIL Image to numpy array for opencv
        img_array = np.array(image)

        # 1. Get BLIP detailed description
        inputs = self.kosmos_processor(image, return_tensors="pt").to(self.device)
        out = self.kosmos.generate(**inputs)
        kosmos_desc = self.kosmos_processor.decode(out[0], skip_special_tokens=True)
        
        # 2. Detect objects with GroundingDINO
        prompt = "box . container . jar . food . vegetable . fruit"
        boxes, logits, phrases = predict(
            model=self.grounding_dino,
            image=img_array,
            prompt=prompt,
            box_threshold=0.35,
            text_threshold=0.25
        )
        
        # 3. Get SAM segmentation for each detected box
        self.sam_predictor.set_image(img_array)
        masks = []
        for box in boxes:
            masks.append(
                self.sam_predictor.predict(
                    box=box,
                    multimask_output=False
                )[0]
            )
        
        # 4. For each detected object, get CLIP classification
        detected_objects = []
        candidate_labels = [
            "plastic container", "glass jar", "food storage box",
            "green peas", "corn", "pickles", "peppers", "cucumbers",
            "preserved vegetables", "canned food", "stacked containers",
            "open container", "sealed container"
        ]
        
        for i, (box, phrase, logit) in enumerate(zip(boxes, phrases, logits)):
            # Crop image to box region
            x1, y1, x2, y2 = box.astype(int)
            cropped = image.crop((x1, y1, x2, y2))
            
            # Get CLIP predictions for cropped region
            inputs = self.clip_processor(images=cropped, text=candidate_labels, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]
                
            # Get top 3 predictions
            top_probs, top_indices = probs.topk(3)
            classifications = [
                {"label": candidate_labels[idx], "score": prob.item()}
                for prob, idx in zip(top_probs, top_indices)
            ]
            
            detected_objects.append({
                "box": box.tolist(),
                "grounding_phrase": phrase,
                "grounding_confidence": logit.item(),
                "classifications": classifications,
                "mask": masks[i].tolist() if masks else None
            })
        
        # 5. Analyze spatial relationships
        spatial_relationships = []
        for i, obj1 in enumerate(detected_objects):
            for j, obj2 in enumerate(detected_objects):
                if i < j:  # Only compare each pair once
                    rel = self._analyze_spatial_relationship(obj1["box"], obj2["box"])
                    if rel:
                        spatial_relationships.append({
                            "object1_idx": i,
                            "object2_idx": j,
                            "relationship": rel
                        })
        
        return {
            "kosmos_description": kosmos_desc,
            "detected_objects": detected_objects,
            "spatial_relationships": spatial_relationships
        }
    
    def _analyze_spatial_relationship(self, box1: List[float], box2: List[float]) -> str:
        """Analyze spatial relationship between two bounding boxes."""
        def box_center(box):
            return [(box[0] + box[2])/2, (box[1] + box[3])/2]
        
        center1 = box_center(box1)
        center2 = box_center(box2)
        
        # Vertical relationship
        if abs(center1[0] - center2[0]) < 50:  # If roughly aligned vertically
            if center1[1] > center2[1]:
                return "below"
            else:
                return "above"
        # Horizontal relationship
        elif abs(center1[1] - center2[1]) < 50:  # If roughly aligned horizontally
            if center1[0] > center2[0]:
                return "right_of"
            else:
                return "left_of"
        
        return None
"""
Fruit and Vegetable Detection and Labeling System
"""

import os
import json
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np
import colorsys
from tqdm import tqdm

class FruitVegLabeler:
    """Detailed fruit and vegetable detection and labeling."""
    
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        # We will save labeled images directly in results or in a subfolder?
        # User said "store all the results in results folder with the labeled images"
        # I'll put them in results/labeled_images to keep it organized, but I'll also copy them to results if needed.
        # Let's stick to results/labeled_images for now.
        os.makedirs(os.path.join(output_dir, "labeled_images"), exist_ok=True)
        
        # Food item categories focused on fruits and vegetables
        self.food_categories = {
            'vegetables': [
                'chickpeas', 'green peas', 'peas', 'corn', 'carrots', 'carrot sticks',
                'bell peppers', 'red peppers', 'green peppers', 'yellow peppers',
                'chili peppers', 'cucumbers', 'cucumber slices', 'tomatoes', 'cherry tomatoes',
                'onions', 'garlic', 'broccoli', 'cauliflower', 'green beans',
                'celery', 'lettuce', 'spinach', 'cabbage', 'mushrooms', 'potatoes',
                'sweet potatoes', 'zucchini', 'eggplant', 'radish', 'asparagus'
            ],
            'fruits': [
                'apples', 'oranges', 'berries', 'strawberries', 'grapes',
                'bananas', 'lemons', 'limes', 'watermelon', 'melon',
                'pineapple', 'mango', 'peach', 'pear', 'plum', 'kiwi',
                'blueberries', 'raspberries', 'blackberries', 'cherries'
            ],
            'other': [
                'container', 'jar', 'box', 'bag', 'bottle', 'can'
            ]
        }
        
        # Flatten all labels for CLIP
        self.all_food_labels = []
        for category in self.food_categories.values():
            self.all_food_labels.extend(category)
        
        print(f"Loaded {len(self.all_food_labels)} labels")
        
        self.clip_model = None
        self.clip_processor = None
        
    def load_models(self):
        """Load CLIP for classification."""
        print(f"Loading CLIP model on {self.device}...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        print("✓ CLIP loaded")
        
    def detect_regions_grid(self, image, grid_size=(3, 4)):
        """Detect regions using grid-based approach + color segmentation."""
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        regions = []
        
        # 1. Grid-based detection
        rows, cols = grid_size
        cell_h = height // rows
        cell_w = width // cols
        
        for i in range(rows):
            for j in range(cols):
                x1 = j * cell_w
                y1 = i * cell_h
                x2 = min((j + 1) * cell_w, width)
                y2 = min((i + 1) * cell_h, height)
                
                # Check if region has content
                roi = img_array[y1:y2, x1:x2]
                if roi.size > 0 and np.std(roi) > 10:  # Has variation
                    regions.append({
                        'bbox': [x1, y1, x2, y2],
                        'type': 'grid',
                        'area': (x2-x1) * (y2-y1)
                    })
        
        # 2. Color-based segmentation
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for common fruits/veg
        color_ranges = {
            'orange': [(10, 100, 100), (25, 255, 255)],
            'green': [(35, 50, 50), (85, 255, 255)],
            'red': [(0, 100, 100), (10, 255, 255)],
            'yellow': [(25, 100, 100), (35, 255, 255)],
            'purple': [(125, 50, 50), (150, 255, 255)]
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum size
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append({
                        'bbox': [x, y, x+w, y+h],
                        'type': f'color_{color_name}',
                        'area': area
                    })
        
        # Remove overlapping regions (keep larger ones)
        regions = self._remove_overlaps(regions)
        
        return regions
    
    def _remove_overlaps(self, regions, iou_threshold=0.5):
        """Remove overlapping regions."""
        if not regions:
            return []
        
        # Sort by area (largest first)
        regions = sorted(regions, key=lambda x: x['area'], reverse=True)
        
        keep = []
        for i, region1 in enumerate(regions):
            overlap = False
            for region2 in keep:
                iou = self._calculate_iou(region1['bbox'], region2['bbox'])
                if iou > iou_threshold:
                    overlap = True
                    break
            
            if not overlap:
                keep.append(region1)
        
        return keep
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def classify_region(self, image, bbox, top_k=1):
        """Classify what item is in the region using CLIP."""
        x1, y1, x2, y2 = bbox
        
        # Crop region
        cropped = image.crop((x1, y1, x2, y2))
        
        # Classify with CLIP
        inputs = self.clip_processor(
            images=cropped,
            text=self.all_food_labels,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
        
        # Get top predictions
        top_probs, top_indices = probs.topk(top_k)
        
        classifications = [
            {
                'label': self.all_food_labels[idx],
                'confidence': prob.item(),
                'category': self._get_category(self.all_food_labels[idx])
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return classifications
    
    def _get_category(self, label):
        """Get category for a label."""
        for category, items in self.food_categories.items():
            if label in items:
                return category
        return 'other'
    
    def create_visualization(self, image, detected_items, output_path):
        """Create visualization with labels and confidence scores."""
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        for idx, item in enumerate(detected_items):
            bbox = item['bbox']
            
            # Choose color based on category
            category = item['classifications'][0]['category']
            if category == 'vegetables':
                color = (0, 255, 0)  # Green
            elif category == 'fruits':
                color = (0, 165, 255) # Orange
            else:
                color = (200, 200, 200) # Gray
            
            # Draw bounding box
            cv2.rectangle(
                img_bgr,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                2
            )
            
            # Prepare label
            top_label = item['classifications'][0]
            label_text = f"{top_label['label']}"
            conf_text = f"{top_label['confidence']:.2f}"
            
            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_w, text_h), _ = cv2.getTextSize(f"{label_text} {conf_text}", font, font_scale, thickness)
            
            # Label box
            cv2.rectangle(
                img_bgr,
                (bbox[0], bbox[1] - text_h - 10),
                (bbox[0] + text_w + 10, bbox[1]),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                img_bgr,
                f"{label_text} {conf_text}",
                (bbox[0] + 5, bbox[1] - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
            
        # Save
        cv2.imwrite(output_path, img_bgr)
    
    def process_image(self, image_path):
        """Process a single image."""
        image = Image.open(image_path).convert('RGB')
        
        regions = self.detect_regions_grid(image, grid_size=(4, 5))
        
        detected_items = []
        for region in regions:
            classifications = self.classify_region(image, region['bbox'], top_k=1)
            
            # Filter by confidence
            if classifications[0]['confidence'] > 0.2:
                detected_items.append({
                    'bbox': region['bbox'],
                    'classifications': classifications
                })
        
        # Create visualization
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        viz_path = os.path.join(self.output_dir, "labeled_images", f"{base_name}_labeled.jpg")
        self.create_visualization(image, detected_items, viz_path)
        
        return detected_items

def main():
    labeler = FruitVegLabeler(output_dir="results")
    labeler.load_models()
    
    # Input folder - using the one found earlier
    input_folder = "BOX_Dataset_Labels/01_raw_images"
    
    if not os.path.exists(input_folder):
        # Fallback to test_images if the other one doesn't exist
        input_folder = "test_images"
    
    if not os.path.exists(input_folder):
        print("No input images found.")
        return

    print(f"Processing images from {input_folder}...")
    
    image_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    for img_path in tqdm(image_files):
        labeler.process_image(img_path)
        
    print("Done! Results saved in 'results' folder.")

if __name__ == "__main__":
    main()

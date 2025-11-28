import json
from typing import Dict, List, Any, Tuple
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from enhanced_model_loader import EnhancedModelLoader
import colorsys

class EnhancedImageProcessor:
    def __init__(self, output_dir: str = "labeled_output"):
        """Initialize the enhanced image processor."""
        self.output_dir = output_dir
        self.model_loader = EnhancedModelLoader()
        self.ontology = self._load_ontology()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labeled_images"), exist_ok=True)
        
    def _load_ontology(self) -> Dict[str, Any]:
        """Load the label ontology from JSON file using absolute path."""
        ontology_path = os.path.join(os.path.dirname(__file__), 'label_ontology.json')
        with open(ontology_path, 'r') as f:
            return json.load(f)
    
    def _generate_distinct_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n visually distinct colors."""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.9
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(tuple(int(x * 255) for x in rgb))
        return colors
    
    def save_visualization(self, image: Image.Image, results: Dict[str, Any], image_name: str) -> None:
        """
        Save a copy of the image with detailed labels, bounding boxes, and relationships overlaid.
        """
        # Create output directory if it doesn't exist
        images_dir = os.path.join(self.output_dir, "labeled_images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Prepare output path
        base_name = os.path.basename(image_name)
        name_root, _ = os.path.splitext(base_name)
        output_path = os.path.join(images_dir, f"{name_root}_labeled.jpg")
        
        # Convert PIL image to numpy array for OpenCV operations
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Generate distinct colors for different objects
        colors = self._generate_distinct_colors(len(results["detected_objects"]))
        
        # Draw detected objects
        for idx, obj in enumerate(results["detected_objects"]):
            box = obj["box"]
            color = colors[idx]
            
            # Draw bounding box
            cv2.rectangle(
                img_bgr,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color,
                2
            )
            
            # Prepare label text
            top_class = obj["classifications"][0]
            label_text = f"{top_class['label']} ({top_class['score']:.2f})"
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )
            
            # Draw label background
            cv2.rectangle(
                img_bgr,
                (int(box[0]), int(box[1]) - text_height - 10),
                (int(box[0]) + text_width + 10, int(box[1])),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img_bgr,
                label_text,
                (int(box[0]) + 5, int(box[1]) - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        # Draw spatial relationships
        for rel in results["spatial_relationships"]:
            obj1 = results["detected_objects"][rel["object1_idx"]]
            obj2 = results["detected_objects"][rel["object2_idx"]]
            
            # Get centers of boxes
            center1 = (
                int((obj1["box"][0] + obj1["box"][2]) / 2),
                int((obj1["box"][1] + obj1["box"][3]) / 2)
            )
            center2 = (
                int((obj2["box"][0] + obj2["box"][2]) / 2),
                int((obj2["box"][1] + obj2["box"][3]) / 2)
            )
            
            # Draw relationship line
            cv2.line(img_bgr, center1, center2, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Draw relationship text
            mid_point = (
                int((center1[0] + center2[0]) / 2),
                int((center1[1] + center2[1]) / 2)
            )
            cv2.putText(
                img_bgr,
                rel["relationship"],
                mid_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
        
        # Add KOSMOS description at the top
        description = results["kosmos_description"]
        wrapped_desc = self._wrap_text(description, img_bgr.shape[1], font_scale=0.7)
        
        # Create space for description at top
        pad_height = len(wrapped_desc) * 30 + 20
        padded_img = cv2.copyMakeBorder(
            img_bgr,
            pad_height, 0, 0, 0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        # Draw description
        y_pos = 25
        for line in wrapped_desc:
            cv2.putText(
                padded_img,
                line,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            y_pos += 30
        
        # Save the image
        cv2.imwrite(output_path, padded_img)
    
    def _wrap_text(self, text: str, max_width: int, font_scale: float = 1.0) -> List[str]:
        """Wrap text to fit within a given width."""
        words = text.split()
        lines = []
        current_line = []
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for word in words:
            current_line.append(word)
            (text_width, _), _ = cv2.getTextSize(
                " ".join(current_line),
                font,
                font_scale,
                2
            )
            
            if text_width > max_width - 20:  # 20px margin
                if len(current_line) > 1:
                    current_line.pop()
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines
    
    def process_and_save(self, image: Image.Image, image_name: str) -> Dict[str, Any]:
        """
        Process an image through all models and save the results.
        """
        # Process the image through our enhanced model pipeline
        results = self.model_loader.process_image(image)
        
        # Save visualization
        self.save_visualization(image, results, image_name)
        
        # Save detailed JSON output
        json_path = os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(image_name))[0]}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Append to summary file
        summary_path = os.path.join(self.output_dir, "summary_labels.jsonl")
        with open(summary_path, 'a') as f:
            summary_entry = {"image_name": image_name, **results}
            f.write(json.dumps(summary_entry) + "\n")
        
        return results
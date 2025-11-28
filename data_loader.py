from datasets import load_dataset
from PIL import Image
import os
from typing import Dict, List, Any
import json
from tqdm import tqdm

class DatasetLoader:
    def __init__(self, output_dir: str = "labeled_output"):
        """Initialize the dataset loader with output directory path."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.dataset = None
        self.image_paths = []

    def load_dataset(self) -> None:
        """Load the BOX_Dataset from HuggingFace."""
        try:
            self.dataset = load_dataset("codereinforcement/BOX_Dataset")
            print(f"Dataset loaded successfully")
            
            # Print available splits
            print(f"Available splits: {self.dataset.keys()}")
            
            # Get the first available split
            split_name = list(self.dataset.keys())[0]
            print(f"Using split: {split_name}")
            
            # Get all images from the dataset
            self.image_paths = []
            for idx in range(len(self.dataset[split_name])):
                if 'image' in self.dataset[split_name][idx]:
                    image_info = self.dataset[split_name][idx]['image']
                    if hasattr(image_info, 'filename'):
                        self.image_paths.append(image_info.filename)
                    else:
                        self.image_paths.append(f"image_{idx}.jpg")
            
            print(f"Found {len(self.image_paths)} images in the dataset")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise

    def get_image(self, idx: int) -> Image.Image:
        """Get a specific image from the dataset."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # Get the first available split
        split_name = list(self.dataset.keys())[0]
        return self.dataset[split_name][idx]['image']

    def save_json_output(self, image_name: str, labels: Dict[str, Any]) -> None:
        """Save JSON output for a single image."""
        json_path = os.path.join(self.output_dir, f"{os.path.splitext(image_name)[0]}.json")
        with open(json_path, 'w') as f:
            json.dump(labels, f, indent=2)

    def append_to_summary(self, image_name: str, labels: Dict[str, Any]) -> None:
        """Append results to the summary JSONL file."""
        summary_path = os.path.join(self.output_dir, "summary_labels.jsonl")
        with open(summary_path, 'a') as f:
            summary_entry = {"image_name": image_name, **labels}
            f.write(json.dumps(summary_entry) + "\n")

if __name__ == "__main__":
    # Test the dataset loader
    loader = DatasetLoader()
    loader.load_dataset()
    
    # Print dataset info
    print("Dataset info:")
    print(f"Number of images: {len(loader.image_paths)}")
    
    # Test loading first image
    if len(loader.image_paths) > 0:
        test_image = loader.get_image(0)
        print(f"Successfully loaded test image: {test_image.size}")
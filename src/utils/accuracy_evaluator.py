import torch
from typing import Tuple, List
from torchvision import datasets, transforms
import os
import torchvision
from torchvision.datasets import ImageFolder, FashionMNIST
import shutil
import numpy as np
from PIL import Image
import time
import json
import argparse
from src.models.mobilenet import MobileNetWrapper
import sys

class AccuracyEvaluator:
    def __init__(self, val_dataset_path: str = "data/mini_test"):
        self.device = torch.device('cpu')
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Create or load dataset
        if not os.path.exists(val_dataset_path):
            self._create_mini_dataset(val_dataset_path)
        
        self.val_dataset = ImageFolder(
            val_dataset_path,
            transform=self.transform
        )
        
        # Reduce worker number, use smaller batch size
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=8,  # Reduce batch size
            shuffle=False,
            num_workers=1,  # Reduce worker number
            pin_memory=False  # Disable pin_memory
        )
    
    def _prepare_mini_test_dataset(self, dataset, output_path, samples_per_class=50):
        """Prepare mini test dataset"""
        # Clean existing directory
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        
        # Create directory for each class
        for class_idx in range(10):  # FashionMNIST has 10 classes
            class_dir = os.path.join(output_path, f'class_{class_idx}')
            os.makedirs(class_dir, exist_ok=True)
            
            # Get all indices of the class
            indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
            
            # Randomly select samples_per_class samples
            selected_indices = np.random.choice(indices, samples_per_class, replace=False)
            
            # Save selected images
            for i, idx in enumerate(selected_indices):
                img_tensor, _ = dataset[idx]
                # Convert tensor to numpy array, then to PIL image
                img_array = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode='L').convert('RGB')
                img = img.resize((224, 224))
                img.save(os.path.join(class_dir, f'image_{i}.png'))
        
        print(f"Created mini test dataset with {samples_per_class} samples per class")
        
    def evaluate(self, model) -> Tuple[float, float]:
        try:
            print("Starting model evaluation...")
            if not hasattr(model, 'eval'):
                raise AttributeError("Model must have 'eval' method")
            
            model.eval()
            print("Model set to eval mode")
            
            model = model.to(self.device)
            print(f"Model moved to device: {self.device}")
            
            correct_1 = 0
            correct_5 = 0
            total = 0
            
            start_time = time.time()
            batch_times = []
            
            print("Starting evaluation loop...")
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(self.val_loader):
                    try:
                        batch_start = time.time()
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = model(images)
                        _, predicted = outputs.topk(5, 1, True, True)
                        
                        total += labels.size(0)
                        correct_1 += (predicted[:, 0] == labels).sum().item()
                        correct_5 += sum([1 for i, label in enumerate(labels)
                                        if label in predicted[i]])
                        
                        batch_time = (time.time() - batch_start) * 1000
                        batch_times.append(batch_time)
                        print(f"Batch {batch_idx + 1}: Time = {batch_time:.2f}ms")
                        
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {str(e)}")
                        raise
            
            print("Evaluation loop completed")
            
            accuracy_top1 = correct_1 / total * 100
            accuracy_top5 = correct_5 / total * 100
            avg_latency = sum(batch_times) / len(batch_times)
            p95_latency = np.percentile(batch_times, 95)
            
            result = {
                "accuracy": {
                    "top1": float(accuracy_top1),
                    "top5": float(accuracy_top5)
                },
                "latency": {
                    "average_ms": float(avg_latency),
                    "p95_ms": float(p95_latency),
                    "batch_times": [float(t) for t in batch_times]
                }
            }
            
            print("Preparing JSON output...")
            json_str = json.dumps(result)
            print(json_str)
            return accuracy_top1, accuracy_top5
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            error_result = {
                "error": str(e),
                "accuracy": {"top1": 0, "top5": 0},
                "latency": {"average_ms": 0, "p95_ms": 0, "batch_times": []}
            }
            print(json.dumps(error_result))
            return 0, 0

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()
    
    try:
        # Initialize evaluator and model
        evaluator = AccuracyEvaluator()
        model = MobileNetWrapper(pretrained=True)
        
        # Execute evaluation
        accuracy_top1, accuracy_top5 = evaluator.evaluate(model)
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "accuracy": {"top1": 0, "top5": 0},
            "latency": {"average_ms": 0, "p95_ms": 0, "batch_times": []}
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main() 

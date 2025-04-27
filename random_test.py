import os
import random
from ultralytics import YOLO
import cv2
from datetime import datetime
import pandas as pd

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "common_test_images")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
NUM_SAMPLES = 10

def get_random_images(base_dir, num_samples):
    """Get random images from subdirectories while preserving their true labels"""
    all_images = []
    # Walk through all subdirectories
    for root, _, files in os.walk(base_dir):
        true_label = os.path.basename(root)  # The folder name is the true aircraft type
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                all_images.append((image_path, true_label))
    
    return random.sample(all_images, min(num_samples, len(all_images)))

def main():
    # Load the model
    model = YOLO(MODEL_PATH)
    
    # Get random images
    test_images = get_random_images(TEST_DIR, NUM_SAMPLES)
    
    # Prepare results storage
    results = []
    
    print(f"\nTesting {len(test_images)} random images...")
    print("-" * 80)
    
    # Process each image
    for img_path, true_label in test_images:
        # Make prediction
        pred = model.predict(source=img_path, conf=0.25, verbose=False)[0]
        if pred.probs is not None:
            top_class_id = int(pred.probs.top1)
            predicted_label = model.names[top_class_id]
            confidence = pred.probs.data[top_class_id].item()
            
            # Store result
            results.append({
                'Image': os.path.basename(img_path),
                'True Label': true_label,
                'Predicted Label': predicted_label,
                'Confidence': f"{confidence:.2%}",
                'Correct': predicted_label == true_label
            })
            
            # Print immediate feedback
            print(f"Image: {os.path.basename(img_path)}")
            print(f"True Label: {true_label}")
            print(f"Predicted: {predicted_label} ({confidence:.2%})")
            print(f"Correct: {'✓' if predicted_label == true_label else '✗'}")
            print("-" * 80)
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    accuracy = (df['Correct'].sum() / len(df)) * 100
    
    print(f"\nOverall Accuracy: {accuracy:.1f}%")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(BASE_DIR, f"test_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")

if __name__ == "__main__":
    main()
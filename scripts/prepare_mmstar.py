import os
import io
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def prepare_mmstar_data(output_dir="/code/Intervention/data/mmstar"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading MMStar dataset from HuggingFace...")
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    
    # Extract only what we need for training
    data = []
    print("Processing samples...")
    for item in ds:
        image = item.get("image") or item.get("img") or item.get("pixel_values")
        question = item.get("question") or item.get("prompt") or item.get("text")
        answer = item.get("answer") or item.get("label")
        
        if image is None or question is None or answer is None:
            continue
            
        # Format the prompt exactly like we do in evaluation
        prompt = f"{question}\nAnswer with the option letter only."
        
        # Convert PIL Image to raw bytes to adhere to the required Parquet format
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format or 'PNG')
        img_bytes = img_byte_arr.getvalue()
        
        data.append({
            "image": {"bytes": img_bytes},
            "prompt": prompt,
            "label": str(answer).strip().upper()
        })
        
    print(f"Total valid samples: {len(data)}")
    
    # 500 samples for training, 1000 samples for testing
    train_data, test_data = train_test_split(data, train_size=500, test_size=1000, random_state=42)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Save to parquet (which the training script expects)
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    
    print(f"Saved training data to {train_path}")
    print(f"Saved testing data to {test_path}")

if __name__ == "__main__":
    prepare_mmstar_data()

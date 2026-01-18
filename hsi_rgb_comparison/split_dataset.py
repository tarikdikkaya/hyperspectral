import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
import shutil

def split_dataset(input_csv, train_ratio=0.8, seed=42, images_dir=None, output_dir=None):
    """
    Splits a dataset CSV into training and validation sets based on unique image paths.
    Optionally copies images into train/val directories.
    """
    if not os.path.exists(input_csv):
        print(f"Error: File {input_csv} not found.")
        return

    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Get unique images to prevent data leakage (same image in both train and val)
    # We split by image_path, not by annotation row
    images = df['image_path'].unique()
    print(f"Total unique images: {len(images)}")
    
    # Split the images
    train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=seed)
    
    print(f"Training images: {len(train_images)} (Approx {train_ratio*100}%)")
    print(f"Validation images: {len(val_images)} (Approx {(1-train_ratio)*100}%)")
    
    # Filter the original dataframe
    train_df = df[df['image_path'].isin(train_images)]
    val_df = df[df['image_path'].isin(val_images)]
    
    # Construct output filenames
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_path = output_dir
    else:
        # Default to directory of input csv
        base_path = os.path.dirname(os.path.abspath(input_csv))
        
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    train_csv_path = os.path.join(base_path, f"{base_name}_train.csv")
    val_csv_path = os.path.join(base_path, f"{base_name}_val.csv")
    
    print(f"Saving training data to {train_csv_path} ({len(train_df)} annotations)...")
    train_df.to_csv(train_csv_path, index=False)
    
    print(f"Saving validation data to {val_csv_path} ({len(val_df)} annotations)...")
    val_df.to_csv(val_csv_path, index=False)
    
    # Copy images if directory is provided
    if images_dir:
        print(f"\nScanning images in {images_dir}...")
        
        train_img_dir = os.path.join(base_path, "train_images")
        val_img_dir = os.path.join(base_path, "val_images")
        
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        
        def copy_files(file_list, src, dst):
            count = 0
            missing = 0
            for img_file in file_list:
                src_path = os.path.join(src, img_file)
                dst_path = os.path.join(dst, img_file)
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    count += 1
                else:
                    # Optional: Check for simple name mismatches or warn
                    missing += 1
                    # print(f"Warning: {img_file} not found in source directory.")
            
            if missing > 0:
                print(f"Warning: {missing} files were not found in source directory.")
            return count

        print("Copying training images...")
        n_train = copy_files(train_images, images_dir, train_img_dir)
        print(f"Successfully copied {n_train} training images to {train_img_dir}")
        
        print("Copying validation images...")
        n_val = copy_files(val_images, images_dir, val_img_dir)
        print(f"Successfully copied {n_val} validation images to {val_img_dir}")
        
    print("\nDone! You can now use these files for --csv_train/val and --root_train/val.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset CSV into train and val by image.")
    # Defaults to the file we were just working with
    parser.add_argument("--input", type=str, default="benchmark_annotations_hsi.csv", help="Input CSV file path")
    parser.add_argument("--ratio", type=float, default=0.8, help="Training ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--images_dir", type=str, default=None, help="Path to source directory of images to copy them into train/val folders.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory for generated files.")
    
    args = parser.parse_args()
    
    split_dataset(args.input, args.ratio, args.seed, args.images_dir, args.output_dir)

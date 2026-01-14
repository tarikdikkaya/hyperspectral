
import pandas as pd
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
import os
import glob

def main():
    # Paths
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    bands_file = os.path.join(workspace_root, "neon_aop_bands.csv")
    annotations_file = os.path.join(workspace_root, "benchmark_annotations.csv")
    hsi_folder = os.path.join(workspace_root, "Hyperspectral")

    # Load data
    print("Loading csv files...")
    bands_df = pd.read_csv(bands_file)
    annotations_df = pd.read_csv(annotations_file)

    # Prepare training data
    X = []
    y = []

    # Group annotations by image
    grouped_annotations = annotations_df.groupby('image_path')
    
    unique_images = annotations_df['image_path'].unique()
    print(f"Found {len(unique_images)} unique images in annotations.")
    
    processed_count = 0
    
    for image_name, group in grouped_annotations:
        # Construct HSI file path
        # Annotation: ex: JERC_048_2018.tif -> File: JERC_048_2018_hyperspectral.tif
        base_name = image_name.replace('.tif', '')
        hsi_filename = f"{base_name}_hyperspectral.tif"
        hsi_path = os.path.join(hsi_folder, hsi_filename)
        
        if not os.path.exists(hsi_path):
            # Try original name just in case
            hsi_path = os.path.join(hsi_folder, image_name)
            if not os.path.exists(hsi_path):
                print(f"Warning: Could not find HSI file for {image_name}. Skipping.")
                continue

        try:
            with rasterio.open(hsi_path) as src:
                # Read all bands (bands, height, width)
                # Note: rasterio reads (bands, y, x)
                hsi_data = src.read()
                bands_count, height, width = hsi_data.shape
                

                # Check if band count matches
                if bands_count != len(bands_df):
                    print(f"Skipping {image_name}: Band count mismatch. Image: {bands_count}, CSV: {len(bands_df)}")
                    continue 
                
                # Create mask for trees (0: background, 1: tree)
                mask = np.zeros((height, width), dtype=np.uint8)
                
                # Mark tree locations
                for _, row in group.iterrows():
                    # Scale coordinates
                    xmin_hsi = int(round(row['xmin'] / 10))
                    ymin_hsi = int(round(row['ymin'] / 10))
                    xmax_hsi = int(round(row['xmax'] / 10))
                    ymax_hsi = int(round(row['ymax'] / 10))
                    
                    # Clip to image boundaries
                    xmin_hsi = max(0, xmin_hsi)
                    ymin_hsi = max(0, ymin_hsi)
                    xmax_hsi = min(width, xmax_hsi)
                    ymax_hsi = min(height, ymax_hsi)
                    
                    # Fill mask
                    mask[ymin_hsi:ymax_hsi, xmin_hsi:xmax_hsi] = 1
                
                # Reshape HSI data for pixel selection: (bands, h, w) -> (h, w, bands)
                hsi_reshaped = np.moveaxis(hsi_data, 0, -1)
                
                # Extract Tree pixels (label 1)
                tree_indices = np.where(mask == 1)
                if len(tree_indices[0]) > 0:
                    tree_pixels = hsi_reshaped[tree_indices]
                    X.extend(tree_pixels)
                    y.extend([1] * len(tree_pixels))
                
                # Extract Background pixels (label 0)
                # Sample a similar number of background pixels to keep classes balanced-ish
                bg_indices = np.where(mask == 0)
                if len(bg_indices[0]) > 0:
                    num_bg_samples = min(len(bg_indices[0]), len(tree_indices[0]) if len(tree_indices[0]) > 0 else 100) # Sample broadly if no trees found? unlikely given iterating group.
                    if num_bg_samples > 0:
                        # Random Sampling
                        chosen_indices = np.random.choice(len(bg_indices[0]), num_bg_samples, replace=False)
                        bg_y_coords = bg_indices[0][chosen_indices]
                        bg_x_coords = bg_indices[1][chosen_indices]
                        
                        bg_pixels = hsi_reshaped[bg_y_coords, bg_x_coords, :]
                        X.extend(bg_pixels)
                        y.extend([0] * len(bg_pixels))
                        
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} images...")
                
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    print(f"Total samples collected: {len(X)}")
    
    if len(X) == 0:
        print("No data collected. Exiting.")
        return

    X = np.array(X)
    y = np.array(y)
    
    # Handle NaN values if any (simple imputation or removal)
    # HSI data might have NaNs or nodata values (-9999).
    # Assuming -9999 is nodata, let's replace with NaN and then drop or impute
    # For simplicity in this demo, we'll replace -9999 with 0 or mean, but typically we define nodata.
    # Let's remove rows with NaNs.
    
    # Check for NaNs or Infinite
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    print("Calculating feature importance...")
    importances = rf.feature_importances_
    
    # Create a DataFrame for results
    # Ensure bands_df matches the number of features
    if len(bands_df) != X.shape[1]: 
        print(f"Warning: Number of bands in CSV ({len(bands_df)}) does not match HSI bands ({X.shape[1]}). Using indices.")
        band_names = [f"Band_{i+1}" for i in range(X.shape[1])]
    else:
        band_names = bands_df['BandName'].tolist() # Or combine with nanometer
        
        # Add importances to bands_df if sizes match
        bands_df['importance'] = importances
    
    # If sizes didn't match, we create a new DF
    if 'importance' not in bands_df.columns:
         results_df = pd.DataFrame({'Band': band_names, 'importance': importances})
    else:
         results_df = bands_df[['BandName', 'nanometer', 'importance']]
         
    # Sort by importance
    results_df = results_df.sort_values(by='importance', ascending=False)
    
    print("\nTop 10 Most Important Bands:")
    print(results_df.head(10))
    
    # Save results
    results_path = os.path.join(workspace_root, "hsi_band_importance_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()

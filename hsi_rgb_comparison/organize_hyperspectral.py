import csv
import os
import shutil

csv_path = 'benchmark_annotations.csv'
source_dir = 'Hyperspectral'
dest_dir = 'hyperspectral_selected'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
    print(f"Created directory: {dest_dir}")

unique_images = set()

try:
    with open(csv_path, 'r', encoding='utf-8-sig') as f: # utf-8-sig to handle BOM if present
        reader = csv.DictReader(f)
        for row in reader:
            if 'image_path' in row and row['image_path']:
                unique_images.add(row['image_path'])
            elif ',' in row: # Fallback if headers fail or something
                 pass
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

print(f"Found {len(unique_images)} unique images in CSV.")

count = 0
missing = []

for img_name in unique_images:
    # Construct filename: name.tif -> name_hyperspectral.tif
    root, ext = os.path.splitext(img_name)
    target_filename = f"{root}_hyperspectral{ext}"
    
    src_path = os.path.join(source_dir, target_filename)
    dst_path = os.path.join(dest_dir, target_filename)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        count += 1
    else:
        missing.append(target_filename)

print(f"Successfully copied {count} files to '{dest_dir}'.")

if missing:
    print(f"Warning: {len(missing)} files were not found in '{source_dir}'.")
    print("First 5 missing files:")
    for m in missing[:5]:
        print(f" - {m}")

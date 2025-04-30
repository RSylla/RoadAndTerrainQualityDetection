import os
import shutil
from pathlib import Path

# Define paths
yolo_images_path = 'dataset/images'  # Main images folder containing train/val/test
xml_source_path = 'dataset/xml_annotations'  # Source XML folder
xml_dest_base_path = 'dataset/xml'  # Destination XML folder with train/val/test structure

# Create destination directories if they don't exist
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(xml_dest_base_path, split), exist_ok=True)

# Process each split (train/val/test)
for split in ['train', 'val', 'test']:
    # Get all image files in current split
    image_files = [f.stem for f in Path(os.path.join(yolo_images_path, split)).glob('*') 
                   if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    # Copy corresponding XML files
    for img_name in image_files:
        xml_source = os.path.join(xml_source_path, f"{img_name}.xml")
        xml_dest = os.path.join(xml_dest_base_path, split, f"{img_name}.xml")
        
        if os.path.exists(xml_source):
            shutil.copy2(xml_source, xml_dest)
        else:
            print(f"Warning: XML file not found for {img_name}")

print("XML files distribution completed!")
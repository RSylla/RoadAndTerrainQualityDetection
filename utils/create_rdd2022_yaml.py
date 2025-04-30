import os
import random
from sklearn.model_selection import train_test_split
from ruamel.yaml import YAML

# Set random seed for reproducibility
random.seed(42)

# Define base path
BASE_PATH = "../data/RDD2022/RDD2022_all_countries/"  # Update with your base path
citys_folders = os.listdir(BASE_PATH)
# Function to get all image and annotation pairs
def get_data_pairs():
    data_pairs = []
    # print(citys_folders)
    # Walk through all subdirectories
    for folder in citys_folders:
        images_path = os.path.join(BASE_PATH, f'{folder}/{folder}/train/images')
        print(images_path)
        annotations_path = os.path.join(BASE_PATH, f'{folder}/{folder}/train/labels')
        if os.path.exists(images_path) and os.path.exists(annotations_path):
            for image in os.listdir(images_path):           
                # Get image path (absolute path)
                image_path = os.path.abspath(os.path.join(BASE_PATH, folder, folder, 'train/images', image))
                
                # Get corresponding annotation path
                ann_path = os.path.join(annotations_path, image.replace('.jpg', '.txt'))
                
                # Only add if both image and annotation exist
                if os.path.exists(ann_path):
                    data_pairs.append(image_path)
    
    return data_pairs

# Create data configuration
data = {
    'path': None,   # Same as BASE_PATH
    'train': '',  # Will be set after splitting
    'val': '',    # Will be set after splitting
    'names': [
        'D00_Longitudinal_crack', 
        'D10_Transverse_crack', 
        'D20_Alligator_crack_Partial_pavement', 
        'D40_Pothole_Bump_Rutting_and_other',
        ],  
    'nc': 4  # number of classes
}

# Get all valid image-annotation pairs
data_pairs = get_data_pairs()
print(f"Found {len(data_pairs)} image-annotation pairs")

# Split dataset
train_pairs, val_pairs = train_test_split(data_pairs, test_size=0.2, random_state=42)

# Create train and val txt files
with open(f'{BASE_PATH}train.txt', 'w') as f:
    for img_path in train_pairs:
        f.write(img_path + '\n')

with open(f'{BASE_PATH}val.txt', 'w') as f:
    for img_path in val_pairs:
        f.write(img_path + '\n')

# Update data configuration
data['train'] = 'train.txt'
data['val'] = 'val.txt'

# Save data configuration
with open(f'{BASE_PATH}rdd2022yolo.yaml', 'w') as f:
    YAML().dump(data, f)
    
print("Data configuration saved successfully")

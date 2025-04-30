import xml.etree.ElementTree as ET
import os

def convert_voc_to_yolo(voc_file, yolo_file, label_map=None):
    tree = ET.parse(voc_file)
    root = tree.getroot()
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)
    
    with open(yolo_file, "w") as yolo:
        for obj in root.findall("object"):
            name = obj.find("name").text
            if label_map is not None and name not in label_map:
                if name == "D01":
                    name = "D00"
                elif name == "D11":
                    name = "D10"
                else:
                    name = "D40"
            bndbox = obj.find("bndbox")
            x_min = int(float(bndbox.find("xmin").text))
            y_min = int(float(bndbox.find("ymin").text))
            x_max = int(float(bndbox.find("xmax").text))
            y_max = int(float(bndbox.find("ymax").text))
            
            index = label_map[name] if label_map is not None else name
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            bbox_width = (x_max - x_min) / img_width
            bbox_height = (y_max - y_min) / img_height
            
            yolo.write(f"{index} {x_center} {y_center} {bbox_width} {bbox_height}\n")

            
label_map = {
    'D00': 0, #Linear crack, longitudinal, wheel mark part
    'D10': 1, #Linear crack, lateral, equal interval
    'D20': 2, #Alligator crack
    'D40': 3, #Rutting, pothole, separation
    }
            
data_path = "../data/RDD2022/RDD2022_all_countries/"
citys_folders = os.listdir(data_path)
print(citys_folders)
for folder in citys_folders:
    xml_files = os.listdir(os.path.join(data_path, f'{folder}/{folder}/train/annotations/xmls'))
    # print(xml_files)
    os.makedirs(os.path.join(data_path, f'{folder}/{folder}/train/labels'), exist_ok=True)
    for xml_file in xml_files:
        voc_file = os.path.join(data_path, f'{folder}/{folder}/train/annotations/xmls/{xml_file}')        
        yolo_file = os.path.join(data_path, f'{folder}/{folder}/train/labels/{xml_file.replace(".xml", ".txt")}')
        # print(yolo_file)
        convert_voc_to_yolo(voc_file, yolo_file, label_map)
        print(f"Converted {voc_file} to {yolo_file}")
print("Done!")
                

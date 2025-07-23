# COCO-CONVERTER

# COCO Converter

This project provides a Python script to convert datasets (organized as folders or CSV files) into the COCO annotation format, with optional YOLO-based object detection for generating bounding box annotations.

---

## Features

- Convert a folder-structured dataset (images organized in class-named subfolders) into COCO format JSON.
- Convert CSV-based datasets (with image paths and labels) into COCO format JSON.
- Automatically run YOLOv5 detection on images to generate bounding box annotations.
- Combine YOLO detections with dataset info to produce complete COCO-style annotations.
- Provide a PyTorch Dataset class (`CocoDataset`) to load and use the generated COCO dataset easily.

---

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- pandas
- PIL (Pillow)
- YOLOv5 weights (e.g., `yolov5s.pt`) 

---

## Usage

Run the conversion script from the command line:

```bash
python coco_converter/full_script.py --data_dir PATH_TO_DATA --output OUTPUT_JSON


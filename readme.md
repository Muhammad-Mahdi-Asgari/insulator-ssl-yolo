# Insulator Detection: Baseline vs SSL-Trained Model

This repository contains:
- A baseline YOLO model trained in Summer 2024
- A newly trained model using a semi-supervised pipeline
- Scripts for image and video inference
- A dataset ratio utility

## Repo Structure
(put the tree I wrote earlier)

## Requirements
Python 3.10+
pip install ultralytics opencv-python numpy torch

## Running Image Inference
python image_inference_script/inference_tiled.py \
  --weights models/new_model/weights/best.pt \
  --source data/test_images \
  --save_dir outputs/images

## Running Video Inference
python video_inference_script/tiled_video_inference.py \
  --weights models/new_model/weights/best.pt \
  --source data/test_video.mp4 \
  --save_dir outputs/video

## Dataset Ratio Script
python ratio_script/ratio.py --dataset data/insulator_dataset

## Models

### Baseline Model
Location: models/baseline_model/weights  
Description: YOLO model trained Summer 2025 by team.

### New Model
Location: models/new_model/weights  
Description: YOLO11m model fine-tuned on expanded dataset with added negatives and tiling.
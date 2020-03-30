# nd131 - Intel® Edge AI for IoT Developers Nanodegree Program

## Deploy a People Counter App at the Edge

### Classifier Results Overview

I have tested the code with various models on just Udacity's Project Workspace, so Hardware remains same -  

Below are the results from individual training sessions.

| Model Name with Version (if applicable) | Model Description | Link to Download| Intel Pre-Trained (Y/N) | Inference | Proper functionality replicated with Probability Threshold | Issue (if any)
| -------------------- | ------- | -------------------- | ----- | ---------- | ---------- | ----- |
| <div align="center"> pedestrian-detection-adas-0002 | <div align="center">SSD framework with tuned MobileNet v1 as a feature extractor | <div align="center">Downloaded via Model Optimizer | <div align="center">Y | <div align="center">52-57 ms |  <div align="center">0.9 | <div align="center">NA |
| <div align="center"> person-detection-retail-0013 | <div align="center">MobileNetV2-like backbone that includes depth-wise convolutions | <div align="center">Downloaded via Model Optimizer | <div align="center">Y | <div align="center">44-48 ms |  <div align="center">0.4 | <div align="center">NA |
| <div align="center"> MobileNet v2 COCO model 2018_03_29 | <div align="center">Plain MobileNetV2 | <div align="center">[Link](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) | <div align="center">N | <div align="center">68-71 ms |  <div align="center">NA |<div align="center"> Single person being detected and counted multiple times |

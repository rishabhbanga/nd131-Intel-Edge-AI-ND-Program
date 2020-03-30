# nd131 - Intel® Edge AI for IoT Developers ND Program

## Deploy a People Counter App at the Edge

### Setup

Run the setupEnv.sh script to setup your Environment with all the depenencies needed to run this project, including downloading of person-detection-retail-0013 and pedestrian-detection-adas-0002 pre-trained models using Model Optimizer.

Now open 4 terminal windows and for each terminal run one each of the steps mentioned below:

#### Step 1 - Start the Mosca server

```
cd webservice/server/node-server
node ./server.js
```

You should see the following message, if successful:
```
Mosca server started.
```

#### Step 2 - Start the GUI
Open new terminal and run below commands.

```
cd webservice/ui
npm run dev
```
You should see the following message in the terminal:
```
webpack: Compiled successfully
```

#### Step 3 - FFmpeg Server
Open new terminal and run the below commands.

```
sudo ffserver -f ./ffmpeg/server.conf
```

#### Step 4 - Run the code
Open a new terminal to run the code.
```
python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m model path -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.8 -c GREEN | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

To see the output on a web based interface, open the link http://0.0.0.0:3004 in a browser.

### Classifier Results Overview

I have tested the code with various models on just Udacity's Project Workspace, so Hardware remains same -  

Below are the results from individual training sessions.

| Model Name with Version (if applicable) | Model Description | Link to Download| Intel Pre-Trained (Y/N) | Inference | Proper functionality replicated with Probability Threshold | Issue (if any)
| -------------------- | ------- | -------------------- | ----- | ---------- | ---------- | ----- |
| <div align="center"> pedestrian-detection-adas-0002 | <div align="center">SSD framework with tuned MobileNet v1 as a feature extractor | <div align="center">Downloaded via Model Optimizer | <div align="center">Y | <div align="center">52-57 ms |  <div align="center">0.9 | <div align="center">NA |
| <div align="center"> person-detection-retail-0013 | <div align="center">MobileNetV2-like backbone that includes depth-wise convolutions | <div align="center">Downloaded via Model Optimizer | <div align="center">Y | <div align="center">44-48 ms |  <div align="center">0.4 | <div align="center">NA |
| <div align="center"> MobileNet v2 COCO model 2018_03_29 | <div align="center">Plain MobileNetV2 | <div align="center">[Link](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) | <div align="center">N | <div align="center">68-71 ms |  <div align="center">NA |<div align="center"> Single person being detected and counted multiple times |

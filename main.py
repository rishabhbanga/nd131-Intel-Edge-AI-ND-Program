import argparse
import os
import cv2
import time
import json
import socket
import sys

import logging as log
import paho.mqtt.client as mqtt

from inference import Network
from sys import platform

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    l_desc = "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl."
    d_desc = "The device name, if not 'CPU'"
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    pt_desc = "The probability to use with the bounding boxes"

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument("-m", help=m_desc, required=True)
    required.add_argument("-i", help=i_desc, required=True)
    optional.add_argument("-l", help=l_desc, default='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so')
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='BLUE')
    optional.add_argument("-pt", help=pt_desc, default=0.5)
    args = parser.parse_args()

    return args

def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    current_count = 0
    for box in result[0][0]:
        if box[2] > args.pt:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, 1)
            current_count = current_count + 1
    return frame, current_count

def get_codec(cpu_ext):
    # Get correct CODEC
    if cpu_ext == "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so":
        codec = 0x00000021
    elif cpu_ext == "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib":
        codec = cv2.VideoWriter_fourcc('M','J','P','G')
    else:
        print("Unsupported OS.")
        exit(1)
    return codec

def connect_mqtt():
    log.info("Connecting to MQTT client")
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def infer_on_stream(args, client):
    # Initialize variables
    input_loc = args.i
    cpu_extension = args.l
    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    
    # Flag for single images
    single_image_mode = False

    # Convert the args for color and confidence
    args.c = convert_color(args.c)
    args.pt = float(args.pt)

    log.info("Initializing the Inference Engine")
    infer_network = Network()
    log.info("Loading the network model into the IE")
    infer_network.load_model(args.m, cur_request_id, args.d, cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    log.info("Handling Input Stream")
    if input_loc == 'CAM':
        input_stream = 0
        log.error("ERROR! Webcam Mode not supported")
    elif input_loc.endswith('.jpg') or input_loc.endswith('.bmp'):
        single_image_mode = True
        input_stream = input_loc
        log.info("Image Mode Activated")
    else:
        assert os.path.isfile(input_loc), "Specified input file doesn't exist"
        log.info("Video Mode Activated")
        input_stream = input_loc

    # Get and open video capture
    vidcapt = cv2.VideoCapture(input_stream)
    if input_stream:
        vidcapt.open(input_loc)
    if not vidcapt.isOpened():
        log.error("ERROR! Unable to open source")

    # Grab the shape of the input
    width = int(vidcapt.get(3))
    height = int(vidcapt.get(4))

    # Create a video writer for the output video
    CODEC = get_codec(cpu_extension)
    if not single_image_mode:
        vidout = cv2.VideoWriter('vidout.mp4', CODEC, 30, (width,height))
    else:
        vidout = None

    # Process frames until the video ends, or process is exited
    while vidcapt.isOpened():
        # Read the next frame
        flag, frame = vidcapt.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        log.info("Preprocessing frames")
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        log.info("Starting asynchronous inference for specified request")
        inf_start = time.time()
        infer_network.exec_net(cur_request_id, p_frame)

        log.info("Getting output of inference")
        if infer_network.wait(cur_request_id) == 0:
            det_time = time.time() - inf_start
            log.info("Fetching results of the inference request")
            result = infer_network.get_output(cur_request_id)
            log.info("Extracting desired stats from the results")
            frame, current_count = draw_boxes(frame, result, args, width, height)
            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                       cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            # When new person enters the video
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))
                log.info("Calculating duration of person in frame")

            if current_count < last_count:
                duration = int(time.time() - start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count

            if key_pressed == 27:
                break

        log.info("Sending the frame to the FFMPEG server")
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        #Depending on image or video write out frame
        if single_image_mode:
            cv2.imwrite('imgout.jpg', frame)
        else:
            vidout.write(frame)

    # Release the vidout writer, capture, and destroy any OpenCV windows
    log.info("Exiting Program")
    if not single_image_mode:
        vidout.release()
    vidcapt.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    args = get_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)

if __name__ == "__main__":
    main()
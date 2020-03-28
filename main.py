"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
# MQTT_PORT = 1883
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to .jpg/.bmp file or CAM")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    ###############Good to have
    parser.add_argument("-pc", "--perf_counts", type=str, default=False,
                        help="Print performance counters")
    ###############Good to have
    parser.add_argument("-c", "--color", 
                        help="Color of bounding boxes to be drawn; RED, GREEN or BLUE"
                        "(BLUE by default)")
    return parser

###############Good to have
def performance_counts(perf_count):
    """
    print information about layers of the model.
    :param perf_count: Dictionary consists of status of the layers.
    :return: None
    """
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))
###############Good to have

def connect_mqtt():
    log.info("Connecting to MQTT client")
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def convert_color(color_string):
    # Mapping dictionaries back to values , i.e. BLUE to 255,0,0
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        # If invalid dictionary key is provided, i.e. 255,5,0
        return colors['BLUE']

def draw_boxes(frame, result, pt, color, width, height):
    for box in result[0][0]: #Output shape is 1x1x100x7
        conf = box[2]
        if conf >= pt:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
    return frame

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialize variables
    input_loc = args.input
    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    # Set Probability threshold for detections
    prob_threshold = float(args.prob_threshold)
    box_color = convert_color(args.color)

    # Initialise the class
    infer_network = Network()
    infer_network.load_model(args.model, args.device, cur_request_id, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    log.info("Handling Input Stream")
    # Flag for single images
    image_flag = False
    # Check if input is WEBCAM
    if input_loc == 'CAM':
        input_loc = 0
    elif input_loc.endswith('.jpg') or input_loc.endswith('.bmp'):
        image_flag = True

    # Get and open Video Capture
    vidcapt = cv2.VideoCapture(input_loc)
    # Could be video path if input_loc is not set to 0 in case of Webcam
    vidcapt.open(input_loc)

    # Grab the shape of the input
    width = int(vidcapt.get(3))
    height = int(vidcapt.get(4))

    # Create a video writer for output video
    if not image_flag:
        vidout = cv2.VideoWriter('vidout.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 
        30, (width, height))
    else:
        vidout = None

    if not vidcapt.isOpened():
        log.error("ERROR! Unable to open video source")
    log.info("Reading from Video Capture")
    flag, frame= vidcapt.read()
    if not flag:
        return
    key_pressed = cv2.waitKey(60)

    log.info("Preprocessing images")
    pp_frame = cv2.resize(frame, (width, height))
    pp_frame = pp_frame.transpose(2,0,1)
    pp_frame = pp_frame.reshape(1,*pp_frame.shape)

    log.info("Starting asynchronous inference for specified request")
    inf_start = time.time()
    infer_network.exec_net(cur_request_id, pp_frame)

    log.info("Waiting for the result")
    if infer_network.wait(cur_request_id) == 0:
        det_time = time.time() - inf_start
        log.info("Fetching results of the inference request")
        result = infer_network.get_output(cur_request_id)
        frame = draw_boxes(frame, result, prob_threshold, box_color, width, height)
        log.info("Extracting desired stats from the results")
        if args.perf_counts:
            perf_count = infer_network.performance_counter(cur_request_id)
            performance_counts(perf_count)
        frame, current_count = ssd_out(frame, result)
        inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
        cv2.putText(frame, inf_time_message, (15, 15),
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

        # When new person enters the video
        if current_count > last_count:
            start_time = time.time()
            total_count = total_count + current_count - last_count
            client.publish("person", json.dumps({"total": total_count}))
            log.info("Calculating duration of person in the video")

        if current_count < last_count:
            duration = int(time.time() - start_time)
            # Publish messages to the MQTT server
            # Topic "person/duration": key of "duration" ###
            client.publish("person/duration", json.dumps({"duration": duration}))
         
        # Topic "person": keys of "count" and "total" ###
        client.publish("person", json.dumps({"count": current_count}))
        last_count = current_count

        log.info("Sending the frame to the FFMPEG server")
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        #Depending on image or video write out frame
        if image_flag:
            cv2.imwrite('output.jpg', pp_frame)
        else:
            vidout.write(pp_frame)

        if key_pressed == 27: #Key 27 is the Escape button
            return

    log.info("Exiting Program")
    if not image_flag:
        vidout.release()
    vidcapt.release()
    cv2.destroyAllWindows()
    #Disconnect from MQTT
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)

if __name__ == '__main__':
    main()
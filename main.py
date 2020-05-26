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
import numpy as np
import time

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
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
                        help="Path to image or video file")
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
    return parser


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def draw_mask(result, width, height):
    """
    Draw symantic mask on the Person
    """
    classes = cv2.resize(result[0].transpose((1,2,0)), (width, height), interpolation=cv2.INTER_NEAREST )
    unique_classes = np.unique(classes)
    out_mask=classes*(255/90)
    out_mask=np.dstack((out_mask, out_mask, out_mask))
    out_mask=np.uint8(out_mask)
    return out_mask, unique_classes

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    # Initialise the class
    inference_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    ### TODO: Load the model through `infer_network` ###
    inference_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape=inference_network.get_input_shape()
    log.warning("Input Shape"+str(net_input_shape))
    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    current_count = 0
    last_count = 0
    total_count = 0
    missing_frame=0
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
                   
        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
                        

        ### TODO: Start asynchronous inference for specified request ###
        inputs = {
            'image_tensor' : p_frame,
            'image_info': (height, width, 1) } 
        inference_network.exec_net(inputs)

        ### TODO: Wait for the result ###
        if inference_network.wait() == 0:
            
            ### TODO: Get the results of the inference request ###
            result = inference_network.extract_output()
            
            current_count = 0
            for box in result[0][0]: # result shape is 1x1xNx7 where N is detected bounding boxes
                conf = box[2]
                if conf >= prob_threshold:
                    missing_frame=0
                    xmin = int(box[3] * width)
                    ymin = int(box[4] * height)
                    xmax = int(box[5] * width)
                    ymax = int(box[6] * height)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 125, 255), 1)
                    current_count+=1
            
            if current_count==0:
                missing_frame+=1
                log.warning("Missing Frame"+str(missing_frame))
            ### TODO: Extract any desired stats from the results ###
            if current_count>last_count:
                log.warning("Person is entered"+str(current_count)+ " : "+str(last_count))
                start_time = time.time()
                total_count += current_count-last_count
                client.publish('person', json.dumps({"total": total_count}))
            elif current_count<last_count:
                duration = int(time.time()-start_time)
                client.publish('person/duration', json.dumps({"duration": duration}))
            client.publish('person/duration', json.dumps({"count": current_count}))
            last_count = current_count
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        
    #realease capture and destroy all window
    cap.release()
    cv2.destroyAllWindows()
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
    infer_time = time.time()
    # Infer
    infer_on_stream(args, client)
    time_took = time.time() - infer_time
    log.info("Inference Time took"+str(time_took))
if __name__ == '__main__':
    main()

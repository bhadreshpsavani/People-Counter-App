#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device='cpu', cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        ### Load the model ###
        model_xml=model
        model_bin=os.path.splitext(model_xml)[0]+'.bin'
        
        ### Add Plugin###
        self.plugin=IECore()
        
        ### Add CPU Extension, if aplicable
        if cpu_extension and 'CPU' in device:
            self.plugin.add_extension(cpu_extension, device)
        
        ### Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        ### Check supported layer
        supported_layers = self.plugin.query_network(network=self.network, device_name=device)
        
        unsupported_layer =  [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layer)!=0:
            log.error('We got unsupported Layers {}'.format(unsupported_layer));
            log.error('We need to add extension for unsupported layer');
            exit(1);
            
        ### Load IENetwork into the plugin ###
        self.exec_network = self.plugin.load_network(self.network, device)
        
        ### get input and output layer ###
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        return

    def get_input_shape(self):
        """
        get input shape of the network
        """
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, inputs, request_id = 0):
        """
        Makes an asynchronious inference request, given an input request
        """
        return self.exec_network.start_async(request_id= request_id,
                                            inputs={self.input_blob:inputs})

    def wait(self):
        """
        checks the status of inference request
        """
        return self.exec_network.requests[0].wait(-1)

    def extract_output(self):
        """
        returns the list of results of output layer of network
        """
        return self.exec_network.requests[0].outputs[self.output_blob]

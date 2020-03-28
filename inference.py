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
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.self.net_plugin = None
        self.infer_request_handle = None

    def load_model(self, model, device="CPU", num_requests, cpu_extension=None, plugin=None):
    ###############Good to have
    # def load_model(self, model, device="CPU", input_size, output_size, num_requests, cpu_extension=None, plugin=None):
    ###############Good to have
        log.info("Loading model")
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        log.info("Initializing Plugin")
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as an IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        log.info("Loading IR")
 
        if self.plugin.device == "CPU":
            log.info("Checking whether all layers are supported or CPU extension is defined")
            supported_layers = self.plugin.get_supported_layers(self.network)
            not_supported_layers = \
                [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by "
                          "the plugin for specified device {}:\n {}".
                          format(self.plugin.device,
                                 ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path"
                          " in command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)

        if num_requests == 0:
            self.self.net_plugin = self.plugin.load_network(network=self.network)
        else:
            self.self.net_plugin = self.plugin.load_network(network=self.network, num_requests=num_requests)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        ###############Good to have
        assert len(self.net.network.keys()) == input_size, \
            "Supports only {} input topologies".format(len(self.network.inputs))
        assert len(self.network.outputs) == output_size, \
            "Supports only {} output topologies".format(len(self.network.outputs))
        ###############Good to have
        return self.plugin, self.get_input_shape()

    def get_input_shape(self):
        return self.network.inputs[self.input_blob].shape
   
    ###############Good to have
    def performance_counter(self, request_id):
        """
        Queries performance measures per layer to get feedback of what is the
        most time consuming layer.
        :param request_id: Index of Infer request value. Limited to device capabilities
        :return: Performance of the layer  
        """
        perf_count = self.self.net_plugin.requests[request_id].get_perf_counts()
        return perf_count
    ###############Good to have

    def exec_net(self, request_id, image):
        self.infer_request_handle = self.self.net_plugin.start_async(request_id=request_id, 
            inputs={self.input_blob: image})
        return self.infer_request_handle

    def wait(self, request_id):
        status = self.self.net_plugin.requests[request_id].wait(-1)
        # Note - Used -1 to ensure we wait till the request is complete
        return status

    def get_output(self, request_id, output=None):
        if output:
            videoOutput = self.infer_request_handle.outputs[output]
        else:
            videoOutput = self.self.net_plugin.requests[request_id].outputs[self.out_blob]
        return videoOutput
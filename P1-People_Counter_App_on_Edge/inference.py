import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''
    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.net_plugin = None
        self.infer_request = None

    def load_model(self, model, num_requests, device="CPU", cpu_extension=None, plugin=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        log.info("Loading model")
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        log.info("Initializing Plugin")
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        log.info("Loading IR")

        log.info("Checking whether all layers are supported or CPU extension is defined")
        supported_layers = self.plugin.query_network(self.network, device_name=device)
        unsupported_layers = \
            [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

        # Load the IENetwork into the plugin
        if num_requests == 0:
            self.net_plugin = self.plugin.load_network(self.network, device)
        else:
            self.net_plugin = self.plugin.load_network(self.network, num_requests, device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return

    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, request_id, image):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        self.infer_request_handle = self.net_plugin.start_async(request_id=request_id,
            inputs={self.input_blob: image})
        return

    def wait(self, request_id):
        '''
        Checks the status of the inference request.
        '''
        status = self.net_plugin.requests[request_id].wait(-1)
        return status

    def get_output(self, request_id, output=None):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        if output:
            result = self.infer_request_handle.outputs[output]
        else:
            result = self.net_plugin.requests[request_id].outputs[self.output_blob]
        return result
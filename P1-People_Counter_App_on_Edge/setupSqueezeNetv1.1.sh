git clone https://github.com/forresti/SqueezeNet.git
cd SqueezeNet/SqueezeNet_v1.1
echo "Inside 1.1"
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_caffe.py --input_model squeezenet_v1.1.caffemodel --input_proto deploy.prototxt --output_dir /home/workspace/model
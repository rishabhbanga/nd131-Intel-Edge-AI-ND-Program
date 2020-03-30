wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
cd ssd_mobilenet_v2
echo "Inside ssd_mobilenet_v2"
export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer
#Issue 1 - Setup tools not found.
apt-get install python3-setuptools
python $MOD_OPT/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config $MOD_OPT/extensions/front/tf/ssd_v2_support.json --output_dir /home/workspace/model
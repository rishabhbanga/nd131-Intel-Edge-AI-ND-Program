git clone https://github.com/mystic123/tensorflow-yolo-v3.git
cd tensorflow-yolo-v3
echo "Downloading coco.names and weights"
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget https://pjreddie.com/media/files/yolov3.weights
#Issue 1 - No module named PIL. Since PIL is not Redundant and Pillow is backwards compatiable with PIL
sudo pip3 install Pillow
echo "Running converter"
#Issue 2 - Tensorflow not found -,-
sudo pip3 install tensorflow==1.5
sudo python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights
#Issue 3 - Setup tools not found.
apt-get install python3-setuptools
export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer
echo $MOD_OPT
read -p 'Is OpenVINO version 2019R3 or earlier? (y/n)' answer
if [[ "$answer" == "y" ]] || [[ "$answer" == "Y" ]]; then
	python $MOD_OPT/mo_tf.py --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config $MOD_OPT/extensions/front/tf/yolo_v3.json --batch 1 --output_dir /home/workspace/model
else
	python $MOD_OPT/mo_tf.py --input_model frozen_darknet_yolov3_model.pb --transformations_config $MOD_OPT/extensions/front/tf/yolo_v3.json --batch 1 --output_dir /home/workspace/model
fi
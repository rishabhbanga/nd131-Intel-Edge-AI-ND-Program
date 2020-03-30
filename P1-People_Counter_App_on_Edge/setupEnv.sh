pip install requests pyyaml -t /usr/local/lib/python3.5/dist-packages && clear && source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5

curl -sSL https://deb.nodesource.com/gpgkey/nodesource.gpg.key | sudo apt-key add -
 VERSION=node_6.x
 DISTRO="$(lsb_release -s -c)"
echo "deb https://deb.nodesource.com/$VERSION $DISTRO main" | sudo tee /etc/apt/sources.list.d/nodesource.list
echo "deb-src https://deb.nodesource.com/$VERSION $DISTRO main" | sudo tee -a /etc/apt/sources.list.d/nodesource.list
sudo apt-get update
sudo apt-get install nodejs
 
sudo apt update
sudo apt-get install python3-pip
pip3 install numpy
pip3 install paho-mqtt
sudo apt install libzmq3-dev libkrb5-dev
sudo apt install ffmpeg

sudo add-apt-repository python3-opencv
sudo apt-get update

cd /opt/intel/openvino/deployment_tools/tools/model_downloader
sudo ./downloader.py --name person-detection-retail-0013 --output_dir /home/workspace
cd -
python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /home/workspace/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
from hailo_sdk_client import ClientRunner, InferenceContext
import os
import numpy as np
import cv2

hardware = "hailo8"
model_path = "/home/lucas/hdd/yolov8-raspberry-pi/weights/finetuned"

# Translating the ONNX model to HAR
runner = ClientRunner(hw_arch=hardware)

# returns
# hn -> JSON representation of the graph structure of the model
# npz -> weights of the model   
hn, npz = runner.translate_onnx_model(model_path + r'.onnx', 
                                      start_node_names=['images'], 
                                      net_input_shapes=[1, 3, 640, 640],
                                      # Stop the parsing before the DFL + NMS
                                      end_node_names=['/model.22/cv2.0/cv2.0.2/Conv',
                                                      '/model.22/cv3.0/cv3.0.2/Conv',
                                                      '/model.22/cv2.1/cv2.1.2/Conv',
                                                      '/model.22/cv3.1/cv3.1.2/Conv',
                                                      '/model.22/cv2.2/cv2.2.2/Conv',
                                                      '/model.22/cv3.2/cv3.2.2/Conv'])


runner.save_har(f"{model_path}.har")

# Optimizing + Quantizing the HAR model
model_script_commands = [
    "nms_postprocess(meta_arch=yolov8, engine=cpu, nms_scores_th=0.2, nms_iou_th=0.4)\n",
]

runner.load_model_script("".join(model_script_commands))

runner.optimize_full_precision()

# Importing the calibration data
calib_data_path = "/home/lucas/hdd/yolov8-raspberry-pi/calib_data"
files = os.listdir(calib_data_path)
n_files = len(files)
calib_data = np.zeros((n_files, 640, 640, 3))

for i in range(0, n_files):
    img = cv2.imread(f'{calib_data_path}/{files[i]}')
    img = cv2.resize(img, (640, 640))
    calib_data[i, :, :, :] = img

runner.optimize(calib_data)
runner.save_har(f"{model_path}_opti.har")

# Compiling the quantized HAR
hef = runner.compile()

hef_file_name = f"{model_path}.hef"
with open(hef_file_name, "wb") as f:
    f.write(hef)
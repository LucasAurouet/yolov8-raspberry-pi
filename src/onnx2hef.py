from hailo_sdk_client import ClientRunner

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


runner.save_har(model_path + r'.har')

# Optimizing + Quantizing the HAR model

# Compiling the quantized HAR
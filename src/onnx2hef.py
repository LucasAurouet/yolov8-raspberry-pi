from hailo_sdk_client import ClientRunner

hardware = "hailo8"
model_path = "home/lucas/hdd/yolov8-raspebrry-pi/weights/finetuned"

runner = ClientRunner(hw_arch=hardware)

# returns
# hn -> JSON representation of the graph structure of the model
# npz -> weights of the model   
hn, npz = runner.translate_onnx_model(model_path + r'.onnx')

runner.save_har(model_path + r'.har')
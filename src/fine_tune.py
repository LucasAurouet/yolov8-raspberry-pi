from ultralytics import YOLO

def get_model(model_name):

    return YOLO(model_name + '.pt')

def finetune(model, data_path, save_path, epochs, res):

    model.train(data=data_path, epochs=epochs, imgsz=res)
    
    model.save(save_path)
    
    print(f'model weights saved at {save_path}')
    
from ultralytics import YOLO

def run_training(yolo_models):
    # Initialize YOLO model with pretrained weights
    for model_name in yolo_models:
        print(f"Training {model_name}...")
        # Load the model with pretrained weights
        model = YOLO(f'{model_name}.pt')    

        # Train the model on your custom dataset
        results = model.train(
            data='rdd2022.yaml',  # path to your data.yaml file
            epochs=150,                         # number of epochs
            imgsz=640,                          # image size
            batch=4,                            # batch size
            save=True,                          # save results
            device='0',                         # GPU device (use 'cpu' for CPU)
            workers=8,                          # number of workers for data loading
            project=f'runs/train{model_name}',  # directory to save training results
            name=f'exp{model_name}',            # name of the experiment
            lr0=0.01,                           # learning rate
            optimizer='SGD',                    # optimizer type (SGD or Adam)
        )

        # Evaluate the model on validation set
        metrics = model.val()

        # Perform inference on test images
        results = model.predict(
            source='dataset/images/test',   # path to test images
            conf=0.25,                      # confidence threshold
            save=True                       # save results
        )

        # Export the model
        model.export(format='onnx')         # export to ONNX format
        return metrics, results, model

if __name__ == '__main__':
    yolo_models = ['yolov10n', 'yolov10s', 'yolov11n', 'yolov11s', 'yolov12s']
    metrics, results, model = run_training(yolo_models)

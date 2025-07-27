from ultralytics import YOLO
import os

#  Function to load a YOLOv11 model
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    model = YOLO(model_path)
    print(f"✅ Model loaded from: {model_path}")
    return model

# train the YOLO model
def train_model(model, data_path, epochs=10, img_size=640, device="cpu", project="runs/train", name="brain_tumor_detection"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset config not found at: {data_path}")
    
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        device=device,
        project=project,
        name=name
    )
    
    print(f"Training completed. Results saved to: {os.path.join(project, name)}")
    return results

#Example Usage
model_path = "yolo11n.pt" 
data_path = "data.yaml" #Path to your yaml file 

model = load_model(model_path)
results = train_model(model, data_path, epochs=20, device="cuda")

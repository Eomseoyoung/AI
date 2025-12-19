from ultralytics import YOLO

def load_student(device):
    model = YOLO("yolov8n.pt").model
    model.to(device)
    model.train()
    return model

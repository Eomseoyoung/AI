from ultralytics import YOLO

def load_teacher(device):
    model = YOLO("yolov8x.pt").model
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

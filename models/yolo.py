from ultralytics import YOLO

class YOLOModel:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def detect(self, image_path):
        results = self.model(image_path, verbose=False)
        return results[0].boxes
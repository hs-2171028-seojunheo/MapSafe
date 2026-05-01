import numpy as np

class FeatureExtractor:
    def __init__(self):

        self.building_ids = [1, 25, 26, 27]
        self.road_ids = [6, 11, 52]
        self.sky_ids = [2]
        self.green_ids = [4, 9, 17, 34]
        self.wall_ids = [0, 5]

    def extract(self, seg_map, yolo_boxes):
        total_pixels = seg_map.size

        features = {}

        # ✅ Segmentation 비율
        features['building_ratio'] = np.isin(seg_map, self.building_ids).sum() / total_pixels
        features['road_ratio'] = np.isin(seg_map, self.road_ids).sum() / total_pixels
        features['sky_ratio'] = np.isin(seg_map, self.sky_ids).sum() / total_pixels
        features['green_ratio'] = np.isin(seg_map, self.green_ids).sum() / total_pixels
        features['wall_ratio'] = np.isin(seg_map, self.wall_ids).sum() / total_pixels

        # ✅ YOLO 객체 수
        car_count = 0
        truck_count = 0
        person_count = 0

        for box in yolo_boxes:
            cls = int(box.cls)

            if cls == 2:
                car_count += 1
            elif cls == 7:
                truck_count += 1
            elif cls == 0:
                person_count += 1

        features['car_count'] = car_count
        features['truck_count'] = truck_count
        features['person_count'] = person_count

        return features
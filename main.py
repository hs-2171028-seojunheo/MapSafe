import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from models.segformer import SegFormerModel
from models.yolo import YOLOModel
from utils.feature_extractor import FeatureExtractor
from utils.score_model import ScoreModel

IMAGE_DIR = "data/images"

def main():
    print("모델 로딩 중...")

    seg_model = SegFormerModel()
    yolo_model = YOLOModel()
    extractor = FeatureExtractor()

    all_features = []

    image_list = os.listdir(IMAGE_DIR)

    print("이미지 분석 시작...")

    for img_name in tqdm(image_list):
        img_path = os.path.join(IMAGE_DIR, img_name)

        seg_map = seg_model.predict(img_path)
        yolo_boxes = yolo_model.detect(img_path)

        features = extractor.extract(seg_map, yolo_boxes)
        features['image'] = img_name

        all_features.append(features)

    df = pd.DataFrame(all_features)

    # ⭐ 테스트용 점수 (임시)
    raw_score = (
        df['building_ratio'] * 2 +
        df['road_ratio'] * 2 +
        df['green_ratio'] * 1.5 -
        df['wall_ratio'] * 2 -
        df['truck_count'] * 1.5
    )

    df['score'] = 1 + 9 * (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())

    print("AutoGluon 학습 시작...")

    model = ScoreModel()
    model.train(df)

    preds = model.predict(df)

    df['pred_score'] = preds

    print("\n샘플 결과:")
    print(df[['image', 'pred_score']])

    df.to_csv("result.csv", index=False)
    print("\nresult.csv 저장 완료")

if __name__ == "__main__":
    main()
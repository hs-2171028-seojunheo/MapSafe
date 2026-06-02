import os
import shutil
import pandas as pd
import numpy as np
import requests
from pathlib import Path

# 경로에 맞게 수정 필요
from extractors.extractor_yolo import YoloFeatureExtractor
from extractors.extractor_segformer import SegFormerFeatureExtractor
from extractors.extractor_opencv import OpenCVFeatureExtractor
from autogluon.tabular import TabularPredictor
from database.osmid import build_segment_key, normalize_osmid

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INPUT_CSV = "unique_coords_20m.csv"
OUTPUT_CSV = "database/test_db.csv"  # DB에 넣을 최종 완성본
TEMP_IMG_DIR = "temp_4dir_images"    # 임시 이미지 저장 폴더
MODEL_PATH = "models/" 

# 테스트 모드 (전체 돌리려면 None)
TEST_LIMIT = 50

def download_4dir_images(osmid, lat, lng, output_dir):
    """중앙 좌표에서 0, 90, 180, 270도 방향의 사진 4장을 다운받습니다."""
    headings = [0, 90, 180, 270]
    downloaded_paths = []

    os.makedirs(output_dir, exist_ok=True)
    
    for h in headings:
        img_name = f"{osmid}_head{h}.jpg"
        img_path = os.path.join(output_dir, img_name)
        
        if not os.path.exists(img_path):
            url = f"https://maps.googleapis.com/maps/api/streetview?size=640x640&location={lat},{lng}&heading={h}&key={GOOGLE_API_KEY}&return_error_code=true"
            res = requests.get(url)
            if res.status_code == 200:
                with open(img_path, 'wb') as f:
                    f.write(res.content)
                downloaded_paths.append(img_path)
            else:
                print(f"  [경고] {osmid}의 {h}도 사진 다운로드 실패")
        else:
            downloaded_paths.append(img_path)
            
    return downloaded_paths


def clear_temp_dir(temp_dir: str) -> None:
    temp_path = Path(temp_dir)
    if not temp_path.exists():
        return

    for image_file in temp_path.glob("*.jpg"):
        try:
            image_file.unlink()
        except Exception as exc:
            print(f"  [경고] 임시 이미지 삭제 실패: {image_file.name} ({exc})")


def remove_dir(path: str) -> None:
    if not os.path.exists(path):
        return

    try:
        shutil.rmtree(path)
    except Exception as exc:
        print(f"  [경고] 폴더 삭제 실패: {path} ({exc})")

def process_pipeline():
    os.makedirs(TEMP_IMG_DIR, exist_ok=True)
    
    # 1. 대상 좌표 불러오기
    target_df = pd.read_csv(INPUT_CSV)
    required_coord_cols = {
        "osmid",
        "latitude1",
        "longitude1",
        "latitude2",
        "longitude2",
        "latitude3",
        "longitude3",
    }
    if not required_coord_cols.issubset(target_df.columns):
        missing = required_coord_cols.difference(target_df.columns)
        raise ValueError(f"INPUT_CSV에 필요한 컬럼이 없습니다: {sorted(missing)}")

    target_df["osmid"] = target_df["osmid"].map(normalize_osmid)
    target_df["segment_key"] = target_df.apply(
        lambda row: build_segment_key(
            row["osmid"],
            row["latitude1"],
            row["longitude1"],
            row["latitude2"],
            row["longitude2"],
        ),
        axis=1,
    )
    original_row_count = len(target_df)
    target_df = target_df.drop_duplicates(subset=["segment_key"]).reset_index(drop=True)
    duplicate_count = original_row_count - len(target_df)
    if duplicate_count:
        print(f"[System] 중복 방향을 포함한 동일 구간 {duplicate_count}개를 제외합니다.")

    if TEST_LIMIT:
        target_df = target_df.head(TEST_LIMIT)
        print(f"⚠️ [테스트 모드] {TEST_LIMIT}개의 도로 구간만 먼저 테스트합니다.")

    # 2. 모델 및 추출기 로드
    print("[System] 모델 및 AI 추출기를 로드합니다...")
    yolo = YoloFeatureExtractor()
    segformer = SegFormerFeatureExtractor()
    opencv = OpenCVFeatureExtractor()
    predictor = TabularPredictor.load(MODEL_PATH)

    required_features = predictor.feature_metadata_in.get_features()
    
    final_db_rows = []

    for idx, row in target_df.iterrows():
        osmid = row["osmid"]
        segment_key = row["segment_key"]
        start_lat = float(row["latitude1"])
        start_lng = float(row["longitude1"])
        end_lat = float(row["latitude2"])
        end_lng = float(row["longitude2"])
        lat, lng = float(row["latitude3"]), float(row["longitude3"]) # 가운데 좌표 사용
        
        print(f"[{idx+1}/{len(target_df)}] 구간 {segment_key} 분석 중...")
        
        # 구간별 개별 폴더 생성 후 4방향 사진 다운로드
        osmid_dir = os.path.join(TEMP_IMG_DIR, segment_key)
        try:
            img_paths = download_4dir_images(segment_key, lat, lng, osmid_dir)
            if not img_paths:
                continue

            # 4장 사진 각각 특징 추출 (개별 폴더에서 추출)
            try:
                yolo_df = yolo.extract_from_directory(osmid_dir)
                segformer_df = segformer.extract_from_directory(osmid_dir)
                opencv_df = opencv.extract_from_directory(osmid_dir)
            except Exception as exc:
                print(f"  [경고] 구간 {segment_key} 특징 추출 실패: {exc}")
                continue

            extracted_df = yolo_df.merge(segformer_df, on="image_filename", how="inner")
            extracted_df = extracted_df.merge(opencv_df, on="image_filename", how="inner")

            if extracted_df.empty:
                print(f"  [경고] 구간 {segment_key} 병합 결과가 비었습니다.")
                continue

            # 모델 입력 스키마에 맞게 정렬 후 예측
            model_input_df = extracted_df.drop(columns=["image_filename"])
            for col in required_features:
                if col not in model_input_df.columns:
                    model_input_df[col] = 0
            model_input_df = model_input_df[required_features]

            predictions = predictor.predict(model_input_df)
            extracted_df["predicted_score"] = predictions.astype(float)
            extracted_df["predicted_score"] = extracted_df["predicted_score"].clip(1.0, 5.0)

            # 평균 계산 (4방향의 모든 수치를 더해서 4로 나눔)
            averaged_features = extracted_df.drop(columns=["image_filename"]).mean().to_dict()
            averaged_score = float(np.mean(extracted_df["predicted_score"]))
            averaged_score = float(np.clip(averaged_score, 1.0, 5.0))

            # D. DB 스키마에 완벽하게 맞춘 1줄(Row) 데이터 생성
            db_row = {
                "image_filename": f"segment_{segment_key}",
                "osmid": osmid,
                "segment_key": segment_key,
                "latitude": lat,
                "longitude": lng,
                "start_latitude": start_lat,
                "start_longitude": start_lng,
                "end_latitude": end_lat,
                "end_longitude": end_lng,
                "source_dataset": "streetview_4dir",
                "model_name": "YOLO+SegFormer+OpenCV+AutoGluon",
                "split": "full",

                # 피처 (평균값)
                "person_count": round(averaged_features.get("person_count", 0)),
                "car_count": round(averaged_features.get("car_count", 0)),
                "truck_count": round(averaged_features.get("truck_count", 0)),
                "road_ratio": averaged_features.get("road_ratio", 0.0),
                "building_ratio": averaged_features.get("building_ratio", 0.0),
                "wall_ratio": averaged_features.get("wall_ratio", 0.0),
                "vegetation_ratio": averaged_features.get("vegetation_ratio", 0.0),
                "sky_ratio": averaged_features.get("sky_ratio", 0.0),
                "brightness_mean": averaged_features.get("brightness_mean", 0.0),
                "dark_area_ratio": averaged_features.get("dark_area_ratio", 0.0),
                "edge_density": averaged_features.get("edge_density", 0.0),

                # 최종 안전도 점수 (평균값) - 모델 스키마에 따라 두 곳 모두 저장
                "safety_score": averaged_score,
                "predicted_score": averaged_score,
            }

            final_db_rows.append(db_row)
        finally:
            remove_dir(osmid_dir)

    # 3. 최종 CSV 저장
    result_df = pd.DataFrame(final_db_rows)
    column_order = [
        "image_filename",
        "osmid",
        "segment_key",
        "latitude",
        "longitude",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
        "source_dataset",
        "safety_score",
        "person_count",
        "car_count",
        "truck_count",
        "road_ratio",
        "building_ratio",
        "wall_ratio",
        "vegetation_ratio",
        "sky_ratio",
        "brightness_mean",
        "dark_area_ratio",
        "edge_density",
        "model_name",
        "predicted_score",
        "split",
    ]
    result_df = result_df.reindex(columns=column_order)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n분석 완료. DB 적재용 파일 생성됨: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_pipeline()

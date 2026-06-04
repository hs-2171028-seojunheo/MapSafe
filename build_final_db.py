import os
import shutil
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from dotenv import load_dotenv

# 경로에 맞게 수정 필요
from extractors.extractor_yolo import YoloFeatureExtractor
from extractors.extractor_segformer import SegFormerFeatureExtractor
from extractors.extractor_opencv import OpenCVFeatureExtractor
from autogluon.tabular import TabularPredictor
from database.osmid import build_segment_key, normalize_osmid

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INPUT_CSV = "unique_coords_20m.csv"
OUTPUT_CSV = "database/test_db.csv"  # DB에 넣을 최종 완성본
SKIPPED_SEGMENTS_CSV = "database/skipped_segments.csv"
TEMP_IMG_DIR = "temp_4dir_images"    # 임시 이미지 저장 폴더
MODEL_PATH = "models/" 

# 테스트 모드 (전체 돌리려면 None)
TEST_LIMIT = None
RESUME_FROM_OUTPUT = True
CHECKPOINT_EVERY = 100
MAX_CONSECUTIVE_403_SEGMENTS = 5
COLUMN_ORDER = [
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
SKIPPED_COLUMN_ORDER = [
    "segment_key",
    "osmid",
    "latitude",
    "longitude",
    "reason",
]


def get_streetview_metadata(lat, lng):
    params = {
        "location": f"{lat},{lng}",
        "key": GOOGLE_API_KEY,
        "radius": 50,
        "source": "outdoor",
    }
    try:
        res = requests.get(
            "https://maps.googleapis.com/maps/api/streetview/metadata",
            params=params,
            timeout=15,
        )
        data = res.json()
    except requests.RequestException as exc:
        return "REQUEST_ERROR", None, str(exc)
    except ValueError as exc:
        return "METADATA_PARSE_ERROR", None, str(exc)

    return data.get("status", f"HTTP_{res.status_code}"), data.get("pano_id"), data.get("error_message")


def download_4dir_images(osmid, lat, lng, output_dir, pano_id=None):
    """중앙 좌표에서 0, 90, 180, 270도 방향의 사진 4장을 다운받습니다."""
    headings = [0, 90, 180, 270]
    downloaded_paths = []
    failed_statuses = []

    os.makedirs(output_dir, exist_ok=True)
    
    for h in headings:
        img_name = f"{osmid}_head{h}.jpg"
        img_path = os.path.join(output_dir, img_name)
        
        if not os.path.exists(img_path):
            params = {
                "size": "640x640",
                "heading": h,
                "key": GOOGLE_API_KEY,
                "return_error_code": "true",
            }
            if pano_id:
                params["pano"] = pano_id
            else:
                params["location"] = f"{lat},{lng}"

            try:
                res = requests.get(
                    "https://maps.googleapis.com/maps/api/streetview",
                    params=params,
                    timeout=15,
                )
            except requests.RequestException as exc:
                print(f"  [경고] {osmid}의 {h}도 사진 다운로드 실패: {exc}")
                continue

            content_type = res.headers.get("content-type", "")
            if res.status_code == 200 and "image" in content_type:
                with open(img_path, 'wb') as f:
                    f.write(res.content)
                downloaded_paths.append(img_path)
            else:
                failed_statuses.append(res.status_code)
                if "image" in content_type:
                    error_message = "Google Street View가 에러 이미지를 반환했습니다."
                else:
                    error_message = res.text[:300].replace("\n", " ")
                print(
                    f"  [경고] {osmid}의 {h}도 사진 다운로드 실패: "
                    f"HTTP {res.status_code}, content-type={content_type}, "
                    f"응답={error_message}"
                )
        else:
            downloaded_paths.append(img_path)
            
    return downloaded_paths, failed_statuses


def save_results(rows, output_csv):
    result_df = pd.DataFrame(rows)
    result_df = result_df.reindex(columns=COLUMN_ORDER)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')


def save_skipped_segments(rows, output_csv=SKIPPED_SEGMENTS_CSV):
    skipped_df = pd.DataFrame(rows)
    skipped_df = skipped_df.reindex(columns=SKIPPED_COLUMN_ORDER)
    skipped_df = skipped_df.drop_duplicates(subset=["segment_key"], keep="last")
    skipped_df.to_csv(output_csv, index=False, encoding='utf-8-sig')


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
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일에 GOOGLE_API_KEY를 추가하세요."
        )

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

    existing_rows = []
    completed_segment_keys = set()
    skipped_rows = []
    skipped_segment_keys = set()
    if RESUME_FROM_OUTPUT and os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        if "segment_key" in existing_df.columns:
            existing_df = existing_df.drop_duplicates(subset=["segment_key"], keep="last")
            completed_segment_keys = set(existing_df["segment_key"].dropna().astype(str))
            existing_rows = existing_df.reindex(columns=COLUMN_ORDER).to_dict("records")

    if RESUME_FROM_OUTPUT and os.path.exists(SKIPPED_SEGMENTS_CSV):
        skipped_df = pd.read_csv(SKIPPED_SEGMENTS_CSV)
        if "segment_key" in skipped_df.columns:
            skipped_df = skipped_df.drop_duplicates(subset=["segment_key"], keep="last")
            skipped_segment_keys = set(skipped_df["segment_key"].dropna().astype(str))
            skipped_rows = skipped_df.reindex(columns=SKIPPED_COLUMN_ORDER).to_dict("records")

    if RESUME_FROM_OUTPUT:
        already_handled = completed_segment_keys | skipped_segment_keys
        target_df = target_df[~target_df["segment_key"].isin(already_handled)].reset_index(drop=True)
        print(
            f"[System] 기존 결과 {len(existing_rows)}행과 제외 구간 {len(skipped_rows)}개를 보존하고 "
            f"남은 {len(target_df)}개 구간만 이어서 분석합니다."
        )

    # 2. 모델 및 추출기 로드
    print("[System] 모델 및 AI 추출기를 로드합니다...")
    yolo = YoloFeatureExtractor()
    segformer = SegFormerFeatureExtractor()
    opencv = OpenCVFeatureExtractor()
    predictor = TabularPredictor.load(MODEL_PATH)

    required_features = predictor.feature_metadata_in.get_features()
    
    final_db_rows = existing_rows
    new_success_count = 0
    consecutive_403_segments = 0

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
            metadata_status, pano_id, metadata_error = get_streetview_metadata(lat, lng)
            if metadata_status != "OK":
                if metadata_status == "ZERO_RESULTS":
                    print(f"  [정보] 구간 {segment_key} 주변 50m 내 Street View 없음: ZERO_RESULTS")
                    skipped_rows.append({
                        "segment_key": segment_key,
                        "osmid": osmid,
                        "latitude": lat,
                        "longitude": lng,
                        "reason": metadata_status,
                    })
                    skipped_segment_keys.add(segment_key)
                    save_skipped_segments(skipped_rows)
                    consecutive_403_segments = 0
                    continue

                if metadata_status in {"REQUEST_DENIED", "OVER_QUERY_LIMIT"}:
                    save_results(final_db_rows, OUTPUT_CSV)
                    raise RuntimeError(
                        f"Street View Metadata API 오류: {metadata_status}. "
                        f"{metadata_error or ''} API 키/결제/쿼터를 확인하세요. "
                        f"현재까지 {len(final_db_rows)}행을 {OUTPUT_CSV}에 저장했습니다."
                    )

                print(f"  [경고] 구간 {segment_key} Street View metadata 확인 실패: {metadata_status}")
                continue

            img_paths, failed_statuses = download_4dir_images(segment_key, lat, lng, osmid_dir, pano_id)
            if not img_paths:
                if failed_statuses and all(status == 403 for status in failed_statuses):
                    consecutive_403_segments += 1
                    if consecutive_403_segments >= MAX_CONSECUTIVE_403_SEGMENTS:
                        save_results(final_db_rows, OUTPUT_CSV)
                        raise RuntimeError(
                            "Street View API가 여러 구간에서 연속으로 HTTP 403을 반환했습니다. "
                            "일일 사용량/결제 한도/API 권한을 확인한 뒤 다시 실행하세요. "
                            f"현재까지 {len(final_db_rows)}행을 {OUTPUT_CSV}에 저장했습니다."
                        )
                elif failed_statuses and all(status == 404 for status in failed_statuses):
                    print(f"  [정보] 구간 {segment_key} Street View 이미지 없음: STATIC_404")
                    skipped_rows.append({
                        "segment_key": segment_key,
                        "osmid": osmid,
                        "latitude": lat,
                        "longitude": lng,
                        "reason": "STATIC_404",
                    })
                    skipped_segment_keys.add(segment_key)
                    save_skipped_segments(skipped_rows)
                    consecutive_403_segments = 0
                else:
                    consecutive_403_segments = 0
                continue
            consecutive_403_segments = 0

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

            # 4방향 피처를 먼저 평균내어 구간 대표 벡터를 생성
            averaged_features = extracted_df.drop(columns=["image_filename"]).mean().to_dict()
            model_input_df = pd.DataFrame([averaged_features])

            # 모델 입력 스키마에 맞게 정렬 후 구간별로 1회만 예측
            for col in required_features:
                if col not in model_input_df.columns:
                    model_input_df[col] = 0
            model_input_df = model_input_df[required_features]

            predicted_score = float(predictor.predict(model_input_df).iloc[0])
            predicted_score = float(np.clip(predicted_score, 1.0, 5.0))

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

                # 평균 피처 벡터로 1회 예측한 최종 안전도 점수
                "safety_score": predicted_score,
                "predicted_score": predicted_score,
            }

            final_db_rows.append(db_row)
            new_success_count += 1
            if new_success_count % CHECKPOINT_EVERY == 0:
                save_results(final_db_rows, OUTPUT_CSV)
                print(f"  [System] 체크포인트 저장: {len(final_db_rows)}행")
        except KeyboardInterrupt:
            save_results(final_db_rows, OUTPUT_CSV)
            if skipped_rows:
                save_skipped_segments(skipped_rows)
            print(
                f"\n[System] 사용자 중단 감지. 현재까지 {len(final_db_rows)}행을 "
                f"{OUTPUT_CSV}에 저장했습니다."
            )
            raise
        finally:
            remove_dir(osmid_dir)

    # 3. 최종 CSV 저장
    save_results(final_db_rows, OUTPUT_CSV)
    print(f"\n분석 완료. DB 적재용 파일 생성됨: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_pipeline()

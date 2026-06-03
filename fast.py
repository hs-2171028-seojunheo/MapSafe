from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from database.database_setup import SessionLocal
from database.models import SafetyObservation
from database.osmid import osmid_from_image_filename, osmid_image_filename_candidates
import os
import math

import requests
import time
from collections import OrderedDict
from pathlib import Path
from io import BytesIO
from threading import Lock
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

from google import genai
from google.genai import types
from extractors.extractor_yolo import YoloFeatureExtractor
from extractors.extractor_segformer import SegFormerFeatureExtractor
from extractors.extractor_opencv import OpenCVFeatureExtractor
from autogluon.tabular import TabularPredictor

# 1. 환경변수 로드 및 API 키 설정
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            timeout=30_000,
            retry_options=types.HttpRetryOptions(
                attempts=4,
                initial_delay=1.0,
                max_delay=8.0,
                exp_base=2.0,
                jitter=0.5,
                http_status_codes=[408, 500, 502, 503, 504],
            ),
        ),
    )
else:
    print("[Warning] GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")

GEMINI_EXPLANATION_CACHE = OrderedDict()
GEMINI_EXPLANATION_CACHE_LOCK = Lock()
GEMINI_EXPLANATION_CACHE_LIMIT = 256
GEMINI_FALLBACK_CACHE_TTL_SECONDS = 60

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("DEVICE:", DEVICE)

image_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# 4. 서버 시작 시 각종 모델 1회 로드
yolo_extractor = YoloFeatureExtractor(model_path="yolo11n.pt")
segformer_extractor = SegFormerFeatureExtractor(device=None)
opencv_extractor = OpenCVFeatureExtractor()
predictor = TabularPredictor.load("models")

#SHAP 중요도 CSV 데이터 로드 (서버 시작 시 1회만 실행)
shap_csv_path = Path("models/shap_global_importance.csv")
model_rule_text = "별도의 SHAP 분석 데이터가 없습니다. 일반적인 도시 계획 상식을 바탕으로 분석합니다."

if shap_csv_path.exists():
    try:
        df_importance = pd.read_csv(shap_csv_path)
        # 상위 5개의 가장 중요한 변수만 텍스트로 추출
        top_5 = df_importance.head(5)
        rules = []
        for idx, row in top_5.iterrows():
            rules.append(f"- {row['Feature']} (중요도 가중치: {row['Mean_Absolute_SHAP']:.4f})")
        model_rule_text = "\n".join(rules)
        print("[System] ✅ SHAP 중요도 데이터를 성공적으로 로드했습니다.")
        print(f"--- 모델의 주요 판단 기준 ---\n{model_rule_text}\n---------------------------")
    except Exception as e:
        print(f"[System] ❌ CSV 로드 에러: {e}")
else:
    print("[System] ⚠️ models/shap_global_importance.csv 파일이 없어 기본 분석 모드로 동작합니다.")


def get_cached_explanation(prompt: str):
    with GEMINI_EXPLANATION_CACHE_LOCK:
        cached_value = GEMINI_EXPLANATION_CACHE.get(prompt)
        if cached_value is not None:
            explanation, expires_at = cached_value
            if expires_at is not None and expires_at <= time.monotonic():
                del GEMINI_EXPLANATION_CACHE[prompt]
                return None
            GEMINI_EXPLANATION_CACHE.move_to_end(prompt)
            return explanation
        return None


def cache_explanation(prompt: str, explanation: str, ttl_seconds=None):
    expires_at = time.monotonic() + ttl_seconds if ttl_seconds is not None else None
    with GEMINI_EXPLANATION_CACHE_LOCK:
        GEMINI_EXPLANATION_CACHE[prompt] = (explanation, expires_at)
        GEMINI_EXPLANATION_CACHE.move_to_end(prompt)
        while len(GEMINI_EXPLANATION_CACHE) > GEMINI_EXPLANATION_CACHE_LIMIT:
            GEMINI_EXPLANATION_CACHE.popitem(last=False)


def get_numeric_feature(features: dict, key: str, default: float = 0.0) -> float:
    value = features.get(key, default)
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def generate_local_explanation(score: float, features: dict) -> str:
    vegetation = get_numeric_feature(features, "vegetation_ratio")
    brightness = get_numeric_feature(features, "brightness_mean")
    dark_area = get_numeric_feature(features, "dark_area_ratio")
    car_count = int(get_numeric_feature(features, "car_count"))
    truck_count = int(get_numeric_feature(features, "truck_count"))

    environment_text = (
        f"<b>식생 비율</b>이 {vegetation:.1f}%로 확인되어 보행 환경에 쾌적함을 더합니다."
        if vegetation >= 15
        else f"<b>식생 비율</b>이 {vegetation:.1f}%로 높지 않아 녹지 측면의 보완 여지가 있습니다."
    )
    brightness_text = (
        f"<b>어두운 영역</b>이 {dark_area:.1f}%이고 평균 밝기는 {brightness:.1f}로, 조도 환경을 주의해서 살펴볼 필요가 있습니다."
        if dark_area >= 20 or brightness < 90
        else f"<b>어두운 영역</b>은 {dark_area:.1f}%이며 평균 밝기는 {brightness:.1f}로 확인됩니다."
    )
    traffic_text = (
        f"사진에서 일반 차량 {car_count}대와 대형 차량 {truck_count}대가 감지되어 차량 통행에 유의해야 합니다."
        if car_count + truck_count >= 3
        else f"사진에서 일반 차량 {car_count}대와 대형 차량 {truck_count}대가 감지되었습니다."
    )

    return (
        f"본 거리의 안전 점수는 5.0 만점에 {score:.2f}점입니다.<br><br>"
        "<span style=\"color:#7f8c8d;\">상세 AI 리포트 연결이 지연되어 측정값을 기준으로 안내합니다.</span><br>"
        f"{environment_text} {brightness_text} {traffic_text}"
    )


def is_daily_gemini_quota_error(error: Exception) -> bool:
    return "GenerateRequestsPerDayPerProjectPerModel" in str(error)


def request_gemini_explanation(prompt: str):
    for attempt in range(2):
        try:
            return gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=700,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except Exception as error:
            should_retry_rate_limit = (
                getattr(error, "code", None) == 429
                and not is_daily_gemini_quota_error(error)
                and attempt == 0
            )
            if not should_retry_rate_limit:
                raise
            time.sleep(2)


# Gemini API + SHAP 데이터 융합 XAI 리포트 생성 함수
def generate_explanation_with_gemini(score: float, features: dict) -> str:
    """
    추출된 특징, 점수, 그리고 'SHAP CSV 분석 결과'를 모두 바탕으로
    Gemini API를 호출하여 정확한 XAI 리포트를 생성합니다.
    """
    if not gemini_client:
        return generate_local_explanation(score, features)

    prompt = f"""
    당신은 도시 환경 및 보행 안전도를 분석하는 전문가입니다.
    다음은 특정 거리의 사진을 분석하여 도출한 특징 데이터와 최종 예측 점수입니다.

    [현재 사진의 특징 데이터]
    - 보행자 수: {int(features.get('person_count', 0))}명
    - 일반 차량 수: {int(features.get('car_count', 0))}대
    - 대형 트럭/화물차 수: {int(features.get('truck_count', 0))}대
    - 도로 면적 비율: {features.get('road_ratio', 0):.1f}%
    - 건물 면적 비율: {features.get('building_ratio', 0):.1f}%
    - 가로수 및 식생 비율: {features.get('vegetation_ratio', 0):.1f}%
    - 하늘 개방감 비율: {features.get('sky_ratio', 0):.1f}%
    - 막힌 벽/담장 비율: {features.get('wall_ratio', 0):.1f}%
    - 사진 전체 밝기(조도): {features.get('brightness_mean', 0):.1f}
    - 어두운 영역 비율: {features.get('dark_area_ratio', 0):.1f}%
    - 엣지(윤곽선) 밀도: {features.get('edge_density', 0):.1f}%

    [참고할 분석 기준 (내부 데이터)]
    {model_rule_text}
    
    [지시사항]
    1. 분석 브리핑의 첫 문장은 반드시 다음 문장으로만 시작하세요: 
       "본 거리의 안전 점수는 5.0 만점에 {score:.2f}점입니다."
    2. 첫 문장 이후 줄바꿈(<br><br>)을 한 번 하고, 이 거리가 왜 해당 점수를 받았는지 3~4문장 분량으로 분석 이유를 이어서 작성하세요.
    3. [매우 중요] "우리 모델이 가장 중요한 판단 기준으로 삼는", "두 번째로 중요하게 평가하는", "안전도 평가에 유의미한 영향을 미치는", "가중치" 등의 메타적인/기계적인 표현을 절대 사용하지 마세요.
    4. 주어진 기준을 바탕으로 분석하되, 마치 전문가가 거리를 직접 보고 자연스럽게 환경을 묘사하듯이 문장을 구성하세요. (예: "가로수 비율이 20%로 높게 조성되어 있어 쾌적한 보행 환경을 제공합니다. 다만, 대형 트럭이 2대 발견되어...")
    5. 보행자나 차량 등의 개수를 언급할 때는 소수점이 아닌 반드시 정수(자연수)로만 표현하세요. (예: 2.0대 -> 2대)
    6. 응답은 웹사이트에 바로 렌더링할 수 있도록 HTML 태그를 사용해주세요. 문단 구분에 <br>을 활용하고, 강조하고 싶은 핵심 명사나 특징에는 <b> 태그를 사용하세요.
    """

    cached_explanation = get_cached_explanation(prompt)
    if cached_explanation is not None:
        return cached_explanation

    try:
        response = request_gemini_explanation(prompt)
        
        # 마크다운 찌꺼기 제거
        clean_text = (response.text or "").replace("```html", "").replace("```", "").strip()
        if not clean_text:
            raise ValueError("Gemini API가 비어 있는 응답을 반환했습니다.")
        cache_explanation(prompt, clean_text)
        return clean_text
    except Exception as e:
        error_code = getattr(e, "code", "unknown")
        print(f"[Gemini API Error] code={error_code} type={type(e).__name__}: {e}", flush=True)
        fallback_explanation = generate_local_explanation(score, features)
        cache_explanation(prompt, fallback_explanation, ttl_seconds=GEMINI_FALLBACK_CACHE_TTL_SECONDS)
        return fallback_explanation
    
# 7. FastAPI 엔드포인트 설정
@app.get("/")
def root():
    return {"message": "MapSafe FastAPI Server with Gemini XAI"}

# DB 세션 의존성 주입 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/api/safety/all")
def get_all_safety_data(db = Depends(get_db)):
    """
    지도 렌더링용 전체 데이터 조회 API
    프론트엔드의 GeoJSON 도로선에 안전 점수를 매핑하기 위해 호출됩니다.
    """
    observations = db.query(SafetyObservation).all()
    result = []
    for obs in observations:
        result.append({
            "id": obs.id,
            "osmid": obs.osmid or osmid_from_image_filename(obs.image_filename),
            "latitude": obs.latitude,
            "longitude": obs.longitude,
            "start_latitude": obs.start_latitude,
            "start_longitude": obs.start_longitude,
            "end_latitude": obs.end_latitude,
            "end_longitude": obs.end_longitude,
            "predicted_score": obs.predicted_score if obs.predicted_score is not None else obs.safety_score
        })
    return result

def haversine_m(lat1, lng1, lat2, lng2):
    R = 6371000
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])

    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


@app.get("/api/safety/nearby")
def get_nearby_safety_data(lat: float, lng: float, radius: float = 20, db=Depends(get_db)):
    observations = db.query(SafetyObservation).all()

    result = []
    for obs in observations:
        if obs.latitude is None or obs.longitude is None:
            continue

        distance = haversine_m(lat, lng, obs.latitude, obs.longitude)

        if distance <= radius:
            result.append({
                "id": obs.id,
                "latitude": obs.latitude,
                "longitude": obs.longitude,
                "start_latitude": obs.start_latitude,
                "start_longitude": obs.start_longitude,
                "end_latitude": obs.end_latitude,
                "end_longitude": obs.end_longitude,
                "predicted_score": obs.predicted_score if obs.predicted_score is not None else obs.safety_score,
                "distance": distance
            })

    return result

def build_safety_response(obs):
    # SQLAlchemy 객체를 dict로 변환하여 피처 데이터로 활용
    obs_dict = obs.__dict__.copy()
    obs_dict.pop('_sa_instance_state', None)

    score = obs.predicted_score if obs.predicted_score is not None else obs.safety_score
    explanation = generate_explanation_with_gemini(float(score), obs_dict)

    # 프론트엔드 정보창 렌더링 호환을 위해 구글 스트리트뷰 URL 생성
    url = (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size=640x640"
        f"&location={obs.latitude},{obs.longitude}"
        f"&heading=0"
        f"&pitch=0"
        f"&fov=90"
        f"&source=outdoor"
        f"&key={GOOGLE_API_KEY}"
    )

    return {
        "lat": obs.latitude,
        "lng": obs.longitude,
        "safety_score": float(score),
        "explanation": explanation,
        "image_url": url,
        "features": obs_dict
    }

@app.get("/api/safety/observations/{observation_id}")
def get_safety_by_observation_id(observation_id: int, db = Depends(get_db)):
    obs = db.query(SafetyObservation).filter(SafetyObservation.id == observation_id).first()

    if not obs:
        raise HTTPException(status_code=404, detail="해당 도로 구간의 분석 데이터를 찾을 수 없습니다.")

    return build_safety_response(obs)

@app.get("/api/safety/{osmid}")
def get_safety_by_osmid(osmid: str, db = Depends(get_db)):
    """
    캐싱된 도로 초고속 상세 조회 API
    무거운 비전 AI 분석 과정을 생략하고, DB에 캐싱된 피처와 점수를 바탕으로 Gemini 설명만 실시간 생성합니다.
    """
    target_filenames = osmid_image_filename_candidates(osmid)
    obs = db.query(SafetyObservation).filter(SafetyObservation.image_filename.in_(target_filenames)).first()

    if not obs:
        raise HTTPException(status_code=404, detail="해당 osmid의 분석 데이터를 찾을 수 없습니다.")

    return build_safety_response(obs)

@app.get("/predict")
def predict(lat: float, lng: float, heading: int = 0):
    temp_dir = Path("./temp-images")
    temp_dir.mkdir(exist_ok=True)
    image_name = "streetview.jpg"
    image_path = temp_dir / image_name

    url = (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size=640x640"
        f"&location={lat},{lng}"
        f"&heading={heading}"
        f"&pitch=0"
        f"&fov=90"
        f"&source=outdoor"
        f"&key={GOOGLE_API_KEY}"
    )

    response = requests.get(url)
    if "image" not in response.headers.get("content-type", ""):
        return {
            "error": "Street View image request failed",
            "status_code": response.status_code,
            "message": response.text[:300]
        }

    try:
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(image_path)

        yolo_df = yolo_extractor.extract_from_directory(str(temp_dir))
        segformer_df = segformer_extractor.extract_from_directory(str(temp_dir))
        #print("SEGFORMER DF")
        #print(segformer_df)
        opencv_df = opencv_extractor.extract_from_directory(str(temp_dir))

        # 3. feature 병합
        features_df = yolo_df.merge(segformer_df, on="image_filename", how="inner")
        features_df = features_df.merge(opencv_df, on="image_filename", how="inner")
        
        #print("FEATURES DF")
        #print(features_df)

        input_df = features_df.drop(columns=["image_filename"])
        required_cols = predictor.feature_metadata_in.get_features()
        
        for col in required_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[required_cols]

        score = predictor.predict(input_df).iloc[0]
        score = max(1.0, min(5.0, float(score)))
        feature_dict = input_df.iloc[0].to_dict()
        explanation = generate_explanation_with_gemini(float(score), feature_dict)

        return {
            "lat": lat,
            "lng": lng,
            "safety_score": float(score),
            "explanation": explanation,
            "image_url": url,
            "features": feature_dict
        }
    
    finally:
        if image_path.exists():
            image_path.unlink()

@app.post("/predict-upload")
async def predict_upload(file: UploadFile = File(...)):
    temp_dir = Path("./temp-images")
    temp_dir.mkdir(exist_ok=True)
    image_path = temp_dir / file.filename

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image.save(image_path)

        yolo_df = yolo_extractor.extract_from_directory(str(temp_dir))
        segformer_df = segformer_extractor.extract_from_directory(str(temp_dir))
        opencv_df = opencv_extractor.extract_from_directory(str(temp_dir))

        features_df = yolo_df.merge(segformer_df, on="image_filename", how="inner")
        features_df = features_df.merge(opencv_df, on="image_filename", how="inner")

        input_df = features_df.drop(columns=["image_filename"])
        required_cols = predictor.feature_metadata_in.get_features()

        for col in required_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[required_cols]

        score = predictor.predict(input_df).iloc[0]
        feature_dict = input_df.iloc[0].to_dict()
        score = max(1.0, min(5.0, float(score)))
        # Gemini 호출
        explanation = generate_explanation_with_gemini(float(score), feature_dict)
        return {
            "safety_score": float(score),
            "explanation": explanation,
            "features": feature_dict
        }

    finally:
        if image_path.exists():
            image_path.unlink()

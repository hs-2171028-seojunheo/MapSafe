from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms

from google import genai

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
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("[Warning] GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")

# 2. FastAPI 초기화 및 CORS 설정
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# 3. 모델 로드 관련 함수 및 클래스
image_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

class PerceptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "resnet18",
            pretrained=False
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return self.backbone(x)

def load_perception_model(path):
    model = torch.load(path, map_location=DEVICE, weights_only=False)
    model.to(DEVICE)
    model.eval()
    return model

# 4. 서버 시작 시 각종 모델 1회 로드
yolo_extractor = YoloFeatureExtractor(model_path="yolo11n.pt")
segformer_extractor = SegFormerFeatureExtractor(device=None)
opencv_extractor = OpenCVFeatureExtractor()
safety_model = load_perception_model("perception_models/safety.pth")
lively_model = load_perception_model("perception_models/lively.pth")
wealthy_model = load_perception_model("perception_models/wealthy.pth")
beautiful_model = load_perception_model("perception_models/beautiful.pth")
boring_model = load_perception_model("perception_models/boring.pth")
depressing_model = load_perception_model("perception_models/depressing.pth")
predictor = TabularPredictor.load("models")

# 5. SHAP 중요도 CSV 데이터 로드 (서버 시작 시 1회만 실행)
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

def get_score_from_output(output):
    if output.numel() == 1:
        return output.item()
    probs = torch.softmax(output, dim=1)
    return probs[0, 1].item()

def predict_perception(image):
    img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        safety_output = safety_model(img_tensor)
        lively_output = lively_model(img_tensor)
        wealthy_output = wealthy_model(img_tensor)
        beautiful_output = beautiful_model(img_tensor)
        boring_output = boring_model(img_tensor)
        depressing_output = depressing_model(img_tensor)

        safety = get_score_from_output(safety_output)
        lively = get_score_from_output(lively_output)
        wealthy = get_score_from_output(wealthy_output)
        beautiful = get_score_from_output(beautiful_output)
        boring = get_score_from_output(boring_output)
        depressing = get_score_from_output(depressing_output)

    return {
        "safety": float(safety),
        "lively": float(lively),
        "wealthy": float(wealthy),
        "beautiful" : float(beautiful),
        "boring" : float(boring),
        "depressing" : float(depressing)
    }

# 6. Gemini API + SHAP 데이터 융합 XAI 리포트 생성 함수
def generate_explanation_with_gemini(score: float, features: dict) -> str:
    """
    추출된 특징, 점수, 그리고 'SHAP CSV 분석 결과'를 모두 바탕으로
    Gemini API를 호출하여 정확한 XAI 리포트를 생성합니다.
    """
    if not gemini_client:
        return "Gemini API 키가 설정되지 않아 AI 리포트를 생성할 수 없습니다."

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
    - 시각적 직관적 안전도(Perception): {features.get('safety', 3.0):.2f}점

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

    try:
        response = gemini_client.models.generate_content(
            model='gemini-flash-latest',
            contents=prompt
        )
        
        # 마크다운 찌꺼기 제거
        clean_text = response.text.replace("```html", "").replace("```", "").strip()
        return clean_text
    except Exception as e:
        print(f"[Gemini API Error] {e}")
        return "현재 AI 분석 서버에 일시적인 지연이 발생하여 상세 리포트를 불러올 수 없습니다."
    
# 7. FastAPI 엔드포인트 설정
@app.get("/")
def root():
    return {"message": "MapSafe FastAPI Server with Gemini XAI"}

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
        #perception score 추출
        perception_scores = predict_perception(
            image
        )

        #print("PERCEPTION")
        #print(perception_scores)

        features_df["safety"] = perception_scores[
            "safety"
        ]

        features_df["lively"] = perception_scores[
            "lively"
        ]

        features_df["wealthy"] = perception_scores[
            "wealthy"
        ]
        features_df["beautiful"] = perception_scores[
            "beautiful"
        ]
        features_df["boring"] = perception_scores[
            "boring"
        ]
        features_df["depressing"] = perception_scores[
            "depressing"
        ]
        #print("FEATURES DF")
        #print(features_df)

        input_df = features_df.drop(columns=["image_filename"])
        required_cols = predictor.feature_metadata_in.get_features()
        
        for col in required_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[required_cols]

        score = predictor.predict(input_df).iloc[0]
        feature_dict = input_df.iloc[0].to_dict()
        
        # Gemini 호출
        explanation = generate_explanation_with_gemini(float(score), feature_dict)

        return {
            "lat": lat,
            "lng": lng,
            "safety_score": float(score),
            "explanation": explanation,
            "features": feature_dict
            "image_url": url,
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

        perception_scores = predict_perception(image)
        features_df["safety"] = perception_scores["safety"]
        features_df["lively"] = perception_scores["lively"]
        features_df["wealthy"] = perception_scores["wealthy"]
        features_df["beautiful"] = perception_scores["beautiful"]
        features_df["boring"] = perception_scores["boring"]
        features_df["depressing"] = perception_scores["depressing"]

        input_df = features_df.drop(columns=["image_filename"])
        required_cols = predictor.feature_metadata_in.get_features()

        for col in required_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[required_cols]

        score = predictor.predict(input_df).iloc[0]
        feature_dict = input_df.iloc[0].to_dict()
        
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
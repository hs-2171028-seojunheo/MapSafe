from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import shutil
from torchvision import transforms

from extractors.extractor_yolo import YoloFeatureExtractor
from extractors.extractor_segformer import SegFormerFeatureExtractor
from autogluon.tabular import TabularPredictor

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("DEVICE:", DEVICE)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
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

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            1
        )

    def forward(self, x):

        return self.backbone(x)

def load_perception_model(path):
    model = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    model.to(DEVICE)
    model.eval()

    return model


# 모델은 서버 시작 시 1번만 로드
yolo_extractor = YoloFeatureExtractor(model_path="yolo11n.pt")
segformer_extractor = SegFormerFeatureExtractor(device=None)
safety_model = load_perception_model("perception_models/safety.pth")
lively_model = load_perception_model("perception_models/lively.pth")
wealthy_model = load_perception_model("perception_models/wealthy.pth")
predictor = TabularPredictor.load("models")

def get_score_from_output(output):
    # output shape 예: [1, 2]
    if output.numel() == 1:
        return output.item()

    # 2개 이상이면 softmax 후 positive class 점수 사용
    probs = torch.softmax(output, dim=1)
    return probs[0, 1].item()


def predict_perception(image):
    img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        safety_output = safety_model(img_tensor)
        lively_output = lively_model(img_tensor)
        wealthy_output = wealthy_model(img_tensor)

        safety = get_score_from_output(safety_output)
        lively = get_score_from_output(lively_output)
        wealthy = get_score_from_output(wealthy_output)

    return {
        "safety": float(safety),
        "lively": float(lively),
        "wealthy": float(wealthy)
    }


@app.get("/")
def root():
    return {"message": "MapSafe FastAPI Server"}


@app.get("/predict")
def predict(
    lat: float,
    lng: float,
    heading: int = 0
):
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
        #image.show()

        # 1. YOLO feature 추출
        yolo_df = yolo_extractor.extract_from_directory(str(temp_dir))
        #print("YOLO DF")
        #print(yolo_df)

        # 2. SegFormer feature 추출
        segformer_df = segformer_extractor.extract_from_directory(str(temp_dir))
        #print("SEGFORMER DF")
        #print(segformer_df)

        # 3. feature 병합
        features_df = yolo_df.merge(segformer_df, on="image_filename", how="inner")
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
        #print("FEATURES DF")
        #print(features_df)

        # 4. AutoGluon 입력에서는 image_filename 제거
        input_df = features_df.drop(columns=["image_filename"])
        required_cols = predictor.feature_metadata_in.get_features()
        # 부족한 컬럼 자동 추가
        for col in required_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # 컬럼 순서 맞추기
        input_df = input_df[required_cols]

        #print("INPUT COLUMNS")
        #print(input_df.columns.tolist())

        # 5. 안전 점수 예측
        score = predictor.predict(input_df).iloc[0]

        return {
            "lat": lat,
            "lng": lng,
            "safety_score": float(score),
            "features": input_df.iloc[0].to_dict()
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

        features_df = yolo_df.merge(segformer_df, on="image_filename", how="inner")

        perception_scores = predict_perception(image)

        features_df["safety"] = perception_scores["safety"]
        features_df["lively"] = perception_scores["lively"]
        features_df["wealthy"] = perception_scores["wealthy"]

        input_df = features_df.drop(columns=["image_filename"])

        required_cols = predictor.feature_metadata_in.get_features()

        for col in required_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[required_cols]

        score = predictor.predict(input_df).iloc[0]

        return {
            "safety_score": float(score),
            "features": input_df.iloc[0].to_dict()
        }

    finally:
        if image_path.exists():
            image_path.unlink()

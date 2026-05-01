# 🛡️ MapSafe: 거리 안전도 분석 프로젝트

이미지(스트리트뷰)를 기반으로
**SegFormer + YOLO + AutoGluon**을 활용해
거리의 **안전도 점수(1~10)**를 예측하는 프로젝트입니다.

---

## 📌 프로젝트 개요

* SegFormer → 의미 분할 (건물, 도로, 하늘, 녹지 등)
* YOLO → 객체 탐지 (사람, 차량, 트럭)
* Feature 추출 → 비율 & 개수 기반 데이터 생성
* AutoGluon → 안전도 점수 예측

---

## 🗂️ 프로젝트 구조

```
project/
│
├── data/
│   └── images/          # 분석할 이미지 (필수)
│
├── models/              # SegFormer, YOLO 모델
├── utils/               # feature, score 관련 코드
├── main.py              # 실행 파일
│
└── requirements.txt
```

---

## ⚙️ 실행 환경

* Python: **3.10 또는 3.11 (중요)**
* OS: macOS / Windows 지원

---

## 🚀 설치 방법

```bash
# 1. conda 환경 생성
conda create -n mapsafe python=3.10
conda activate mapsafe

# 2. pip 업데이트
python -m pip install --upgrade pip

# 3. 패키지 설치
pip install -r requirements.txt
```

---

## ▶️ 실행 방법

```bash
conda activate mapsafe
python main.py
```

---

## 📊 결과

실행 후:

* 콘솔 출력: 예측 점수 확인
* CSV 파일 생성 (예: result.csv)

| image    | building_ratio | car_count | score | pred_score |
| -------- | -------------- | --------- | ----- | ---------- |
| img1.jpg | 0.32           | 3         | 7.2   | 7.0        |

---

## 🧠 모델 설명

### 🔹 SegFormer

* ADE20K 기반 semantic segmentation
* 픽셀 단위로 환경 요소 분석

### 🔹 YOLO

* 객체 탐지 (사람, 차량, 트럭)

### 🔹 AutoGluon

* feature 기반 회귀 모델
* 안전도 점수 예측

---

## ⚠️ 주의사항

### ❗ Python 3.13 사용 금지

* AutoGluon + LightGBM 충돌 발생 (segmentation fault)

👉 반드시 Python 3.10 / 3.11 사용

---

## ⭐ 핵심 요약

👉 이미지 → feature → 점수
👉 SegFormer + YOLO + AutoGluon 파이프라인

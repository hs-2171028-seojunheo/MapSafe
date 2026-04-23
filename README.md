# 맵세이프




## 1. 프로젝트 개요

MapSafe는 단일 비전 모델의 한계를 극복하기 위해 객체 탐지 + 의미 분할 + 머신러닝을 결합한 2단계 Hybrid 구조로 설계되었습니다.

이미지에서 다양한 특징을 추출한 뒤, 이를 기반으로 거리 안전도 점수 (1.0 ~ 5.0)를 예측합니다.

특히 메모리(VRAM/RAM) 최적화를 위해 무거운 AI 모델들을 각각 독립된 프로세스(Microservice 구조)로 분리하여, 이미지에서 다양한 특징을 추출한 뒤 최종적으로 거리 안전도 점수 (1.0 ~ 5.0)를 예측합니다.

---

## 2. 파이프라인 구조

### Stage 1: 특징 추출 (Feature Extraction)

- **YOLO11n (객체 탐지)**
  - 사람, 차량, 트럭 개수 추출
  - → `person_count`, `car_count`, `truck_count`

- **SegFormer (의미론적 분할)**
  - 환경 요소 픽셀 비율 계산
  - → `road`, `building`, `wall`, `vegetation`, `sky`

---

### Stage 2: 머신러닝 예측 (Regression)

- **AutoGluon (Tabular Regression)**
  - 총 8개 특징을 입력으로 사용
  - 안전도 점수 예측 모델 학습
  - 데이터 누수 방지를 위해 `image_filename` 제거

---

## 3. 프로젝트 구조

```text
MapSafe/
├── main.py                     # 메인코드 (실행 진입점)
├── extractor_yolo.py           # YOLO 객체 탐지 독립 모듈
├── extractor_segformer.py      # SegFormer 의미 분할 독립 모듈
├── model_predictor.py          # AutoGluon 머신러닝 예측 독립 모듈
├── test.py                     # 테스트 파이프라인
│
├── requirements.txt            # 의존성 패키지
│
├── images/                     # 평가 대상 이미지 폴더
├── ground_truth.csv            # 정답 데이터
│
├── yolo_features.csv           # YOLO 추출 결과 (자동 생성)
├── segformer_features.csv      # SegFormer 추출 결과 (자동 생성)
├── extracted_features.csv      # 두 특징이 병합된 최종 데이터 (자동 생성)
└── models/                     # 학습이 완료된 모델 저장 폴더 (출력)
```

---

## 4. 동작 흐름 (Data Flow)

본 프로젝트는 메모리 해제(OOM 방지)를 위해 main.py가 각 추출기를 subprocess로 호출하여 OS 레벨에서 메모리를 완전히 격리합니다.

1. **입력 데이터 준비**
   - `images/` 폴더에 거리 이미지 저장
   - `ground_truth.csv`에 이미지별 안전도 점수 정의

2. **YOLO 특징 추출**
   - extractor_yolo.py 실행 → yolo_features.csv 산출 후 메모리 완전 해제

3. **SegFormer 특징 추출**
   - extractor_segformer.py 실행 → segformer_features.csv 산출 후 메모리 완전 해제

4. **CSV 특징 병합**
   - 두 CSV 파일을 image_filename 기준으로 Inner Join하여 extracted_features.csv 생성
  
5. **모델 학습**
   - model_predictor.py 실행 → AutoGluon 회귀 모델 학습 후 models/ 에 저장

---

## 5. 필수 데이터 포맷 명세

### 5.1 이미지 폴더 (/images/)
- 평가 대상이 되는 도시 거리 사진을 저장합니다.
- 형식: .jpg, .jpeg, .png
- 최소 해상도: 128x128 이상 권장

### 5.2 정답 데이터 (ground_truth.csv)
모델 학습의 정답지가 되는 파일입니다. images/ 폴더 내의 파일명과 1:1 매칭되어야 합니다.
- 구조 예시
  ```csv
    image_filename,safety_score
    street_01.jpg,4.5
    street_02.jpg,3.2

### 5.3 특징 데이터 (extracted_features.csv)
- main.py 실행 시, 각 독립 모듈에서 생성된 YOLO 결과와 SegFormer 결과를 image_filename을 기준으로 결합(Inner Join)하여 자동 생성되는 중간 데이터 파일입니다.

| 컬럼명 | 타입 | 설명 | 추출 모델 |
|--------|------|------|-----------|
| image_filename | String | 분석 대상 이미지 파일명 (고유 Key) | - |
| person_count | Integer | 이미지 내 감지된 보행자 수 (자연 감시 인자) | YOLO11n |
| car_count | Integer | 이미지 내 감지된 승용차 수 (도로 활력 인자) | YOLO11n |
| truck_count | Integer | 이미지 내 감지된 대형차/트럭 수 (위험/시야 방해 인자) | YOLO11n |
| road_ratio | Float (%) | 전체 이미지 대비 도로 면적 비율 | SegFormer |
| building_ratio | Float (%) | 전체 이미지 대비 건물 면적 비율 (밀도 인자) | SegFormer |
| wall_ratio | Float (%) | 전체 이미지 대비 벽/담장 면적 비율 (폐쇄성 인자) | SegFormer |
| vegetation_ratio | Float (%) | 전체 이미지 대비 식생(나무, 풀) 면적 비율 | SegFormer |
| sky_ratio | Float (%) | 전체 이미지 대비 하늘 면적 비율 (개방감 인자) | SegFormer |

---

## 6. 실행 방법

### 6.1 환경 설정
    # 가상 환경 생성
    python -m venv venv

    # 활성화
    venv\Scripts\activate     # Windows
    source venv/bin/activate  # macOS/Linux

    # 패키지 설치
    pip install -r requirements.txt

### 6.2 테스트 실행
    python test.py
    
### 6.3 전체 파이프라인 실행
    python main.py

---

## 7. 모듈별 개별 사용법 (API)

7.1 이미지 특징만 개별로 추출하기
```python
    from extractor_yolo import YoloFeatureExtractor
    from extractor_segformer import SegFormerFeatureExtractor
    
    # YOLO 추출기 단독 사용
    yolo_ext = YoloFeatureExtractor()
    yolo_df = yolo_ext.extract_from_directory('./images/')
    
    # SegFormer 추출기 단독 사용
    seg_ext = SegFormerFeatureExtractor()
    seg_df = seg_ext.extract_from_directory('./images/')
```

7.2 새로운 데이터 안전도 예측하기
```python
    import pandas as pd
    from model_predictor import SafetyModelPredictor
    
    predictor = SafetyModelPredictor('./extracted_features.csv', './ground_truth.csv', './models/')
    # (학습 완료 가정)
    
    new_features = pd.DataFrame({
        'person_count': [2], 'car_count': [1], 'truck_count': [0],
        'road_ratio': [45.5], 'building_ratio': [28.3], 'wall_ratio': [5.1],
        'vegetation_ratio': [12.2], 'sky_ratio': [9.0]
    })
    
    predictions = predictor.predict(new_features)
    print(f"예측된 안전 점수: {predictions[0]:.2f}")
```
---

## 8. 자주 발생하는 에러 (Troubleshooting)

| 에러 메시지 | 원인 | 해결 방법 |
|------------|------|-----------|
| CUDA out of memory | GPU 메모리 부족 | 개별 모듈에서 `device='cpu'`로 강제 지정 |
| No image files found | 이미지 폴더가 비어 있음 | `images/` 폴더에 `.jpg`, `.png` 이미지 추가 |
| Merge failed / Empty DataFrame | CSV 파일 간 불일치 | 각 CSV 파일에 동일한 `image_filename` 존재 여부 확인 |
| Import fastai failed | 일부 모델 의존성 누락 | 무시 가능 (다른 트리 기반 모델이 자동 사용됨) |
| psutil module not found | 메모리 로깅 라이브러리 누락 | `pip install psutil` 실행 |


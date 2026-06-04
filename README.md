# MapSafe

MapSafe는 거리 이미지와 도로 구간 데이터를 기반으로 보행 환경의 안전도를 1점부터 5점까지 예측하고, 그 결과를 지도와 사진 분석 화면에서 설명 가능한 형태로 보여주는 빅데이터 캡스톤 디자인 프로젝트입니다.

객체 탐지, 의미 분할, OpenCV 기반 시각 특징, AutoGluon 회귀 모델, SHAP 중요도, Gemini 기반 자연어 설명을 결합해 사용자가 특정 도로 구간이 왜 안전하거나 위험하게 평가되었는지 이해할 수 있도록 설계했습니다.

## 핵심 기능

| 기능 | 설명 |
| --- | --- |
| 실시간 도로 분석 | 지도에서 임의 위치를 클릭하면 Google Street View 이미지를 가져와 즉시 안전도를 예측합니다. |
| 안전 지도 서비스 | Kakao Map 위에 성북구 도보 네트워크와 분석된 도로 구간을 색상으로 표시합니다. |
| 도로 구간 상세 분석 | 지도에서 도로 구간을 클릭하면 DB에 캐싱된 4방향 Street View 평균 안전도와 설명을 보여줍니다. |
| 사진 업로드 분석 | 사용자가 업로드한 거리 사진을 분석해 안전 점수와 AI 설명을 제공합니다. |
| 설명 가능한 AI | SHAP 중요도와 추출 특징을 바탕으로 Gemini 또는 로컬 규칙 기반 설명을 생성합니다. |
| DB 기반 캐싱 | 사전 분석된 도로 구간을 SQLAlchemy 모델로 저장해 지도 조회 속도를 높입니다. |

## 전체 구조

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/6f46c07a-d42e-4cd6-9cac-ee0dbdf2d36b" />

## 기술 스택

| 영역 | 사용 기술 |
| --- | --- |
| Frontend | Vue 3, Vite, Vue Router, Kakao Map API |
| Backend | FastAPI, SQLAlchemy, python-dotenv |
| Vision AI | Ultralytics YOLO11n, SegFormer Cityscapes, OpenCV |
| ML/AutoML | AutoGluon Tabular Regression, PyTorch, scikit-learn |
| XAI | SHAP, matplotlib, Gemini API |
| External API | Google Street View Static API, Google Street View Metadata API |
| Data | CSV, GeoJSON, SQLite 또는 DATABASE_URL로 지정한 DB |

## 프로젝트 구조

```text
MapSafe/
├── README.md
├── requirements.txt
├── .gitignore
│
├── main.py                         # FastAPI 서버, 예측 API, XAI 설명 생성
├── pipeline.py                     # 기본 학습 파이프라인: YOLO + SegFormer + AutoGluon
├── model_predictor.py              # AutoGluon 학습, 평가, 예측, SHAP 산출
├── build_final_db.py               # Street View 4방향 분석 후 DB 적재용 CSV 생성
├── Model_01.py                     # ViT 기반 실험 모델 정의
│
├── extractors/
│   ├── extractor_yolo.py           # 사람, 차량, 트럭 객체 수 추출
│   ├── extractor_segformer.py      # 도시 장면 의미 분할 비율 추출
│   └── extractor_opencv.py         # 밝기, 어두운 영역, 엣지 밀도 추출
│
├── database/
│   ├── database_setup.py           # DATABASE_URL 기반 DB 연결 설정
│   ├── models.py                   # safety_observations 테이블 스키마
│   ├── osmid.py                    # OSM ID 및 segment_key 정규화 유틸
│   └── import_observations.py      # CSV를 DB에 적재하는 CLI
│
├── preprocess/
│   └── preprocess_survey.py        # 설문 결과를 ground_truth.csv로 변환
│
└── vue/
    ├── package.json
    ├── vite.config.js
    ├── public/
    │   └── seongbuk_walk.geojson   # 성북구 도보 네트워크 GeoJSON
    └── src/
        ├── App.vue
        ├── main.js
        ├── router/index.js
        ├── components/Menu.vue
        └── views/
            ├── MapView.vue         # 지도, 도로 안전도 오버레이, Roadview
            └── PhotoAnalysisView.vue
```

## 주요 데이터 흐름

### 1. 모델 학습용 데이터 생성

1. `images/` 폴더에 거리 이미지를 준비합니다.
2. 설문 기반 정답 파일 `ground_truth.csv`를 준비합니다.
3. YOLO와 SegFormer가 이미지별 특징 CSV를 생성합니다.
4. `pipeline.py`가 두 CSV를 `image_filename` 기준으로 병합해 `extracted_features.csv`를 만듭니다.
5. `model_predictor.py`가 `extracted_features.csv`와 `ground_truth.csv`를 병합해 AutoGluon 회귀 모델을 학습합니다.
6. 학습 결과는 `models/`에 저장되고, SHAP 결과는 `models/shap_global_importance.csv`, `models/shap_local_values.csv`, `models/shap_summary.png`로 저장됩니다.

### 2. 서비스용 도로 DB 생성

1. `unique_coords_20m.csv`에서 도로 구간의 시작점, 끝점, 중간점 좌표를 읽습니다.
2. `build_final_db.py`가 각 구간의 중간점 기준으로 Street View 4방향 이미지를 다운로드합니다.
3. 각 이미지에서 YOLO, SegFormer, OpenCV 특징을 추출합니다.
4. 4방향 특징을 평균해 한 도로 구간의 대표 특징 벡터를 만듭니다.
5. AutoGluon 모델로 최종 안전도를 예측합니다.
6. `database/test_db.csv`를 생성한 뒤 `database/import_observations.py`로 DB에 적재할 수 있습니다.

### 3. 사용자 서비스 흐름

1. Vue 프론트엔드가 Kakao Map과 Roadview를 렌더링합니다.
2. 사용자가 도보 레이어를 켜면 `/api/safety/all`에서 캐시된 도로 안전도 데이터를 받아옵니다.
3. 도로 구간은 점수 기준으로 위험, 보통, 안전 색상으로 표시됩니다.
4. 사용자가 구간을 클릭하면 `/api/safety/observations/{id}`에서 상세 점수와 설명을 조회합니다.
5. 사용자가 임의 지도 위치를 클릭하면 `/predict`가 Street View 단일 방향 이미지를 실시간 분석합니다.
6. 사용자가 사진을 업로드하면 `/predict-upload`가 업로드 이미지를 실시간 분석합니다.

## 추출 특징

| 특징 | 타입 | 의미 | 추출 모듈 |
| --- | --- | --- | --- |
| `person_count` | Integer | 이미지 내 보행자 수 | YOLO11n |
| `car_count` | Integer | 이미지 내 일반 차량 수 | YOLO11n |
| `truck_count` | Integer | 이미지 내 대형 차량 수 | YOLO11n |
| `road_ratio` | Float | 전체 이미지 대비 도로 픽셀 비율 | SegFormer |
| `building_ratio` | Float | 전체 이미지 대비 건물 픽셀 비율 | SegFormer |
| `wall_ratio` | Float | 전체 이미지 대비 벽/담장 픽셀 비율 | SegFormer |
| `vegetation_ratio` | Float | 전체 이미지 대비 식생 픽셀 비율 | SegFormer |
| `sky_ratio` | Float | 전체 이미지 대비 하늘 픽셀 비율 | SegFormer |
| `brightness_mean` | Float | 흑백 변환 후 평균 밝기 | OpenCV |
| `dark_area_ratio` | Float | 어두운 픽셀 영역 비율 | OpenCV |
| `edge_density` | Float | Canny edge 기준 윤곽선 밀도 | OpenCV |

## 안전 점수 기준

모델은 1.0부터 5.0까지의 연속형 안전 점수를 예측합니다. 지도에서는 프론트엔드 기준에 따라 다음과 같이 표시합니다.

| 점수 구간 | 지도 색상 | 해석 |
| --- | --- | --- |
| 1.0 이상 2.0 미만 | 빨강 | 위험 |
| 2.5 이상 3.5 미만 | 노랑 | 보통 |
| 3.5 이상 5.0 이하 | 초록 | 안전 |

## 설치 및 실행

### 1. Python 환경 준비

AutoGluon과 PyTorch 호환성을 위해 Python 3.10 또는 3.11 사용을 권장합니다.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install fastapi uvicorn sqlalchemy python-dotenv requests pillow google-genai python-multipart openpyxl pandas geopandas shapely psycopg2-binary

```

macOS에서 LightGBM 오류가 발생하면 `libomp`가 필요할 수 있습니다.

```bash
brew install libomp
```

### 2. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
DATABASE_URL=postgresql:///./mapsafe.db
GOOGLE_API_KEY=your_google_streetview_api_key
GEMINI_API_KEY=your_gemini_api_key
```

`GEMINI_API_KEY`가 없으면 FastAPI 서버는 Gemini 호출 대신 로컬 규칙 기반 설명을 반환합니다.

프론트엔드의 Kakao Map 사용을 위해 `vue/.env` 파일을 생성합니다.

```env
VITE_KAKAO_MAP_KEY=your_kakao_javascript_key
```

### 3. 모델 학습

이미지와 정답 파일이 준비되어 있다면 기본 학습 파이프라인을 실행합니다.

```bash
python pipeline.py
```

이 명령은 다음 파일과 폴더를 생성합니다.

| 산출물 | 설명 |
| --- | --- |
| `yolo_features.csv` | YOLO 객체 수 특징 |
| `segformer_features.csv` | SegFormer 의미 분할 비율 |
| `extracted_features.csv` | 학습용 병합 특징 |
| `models/` | AutoGluon 모델 아티팩트 |
| `models/shap_global_importance.csv` | 전역 SHAP 중요도 |
| `models/shap_local_values.csv` | 샘플별 SHAP 값 |
| `models/shap_summary.png` | SHAP 요약 이미지 |

설문 원본 엑셀 파일을 정답 CSV로 변환하려면 다음을 실행합니다.

```bash
python preprocess/preprocess_survey.py
```

### 4. DB 생성 및 적재

서비스 지도에서 빠른 구간 조회를 하려면 사전 분석 DB가 필요합니다.

```bash
python build_final_db.py
python database/import_observations.py --csv database/test_db.csv --source streetview_4dir --split full --replace
```

`build_final_db.py`는 Google Street View API를 사용하므로 `GOOGLE_API_KEY`, 결제 설정, API 사용량 제한을 확인해야 합니다. Street View가 없는 구간은 `database/skipped_segments.csv`에 기록됩니다.

### 5. 백엔드 서버 실행

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

서버가 정상 실행되면 다음 주소에서 상태를 확인할 수 있습니다.

```text
http://127.0.0.1:8000/
```

### 6. 프론트엔드 실행

Node.js는 `vue/package.json` 기준으로 20.19 이상 또는 22.12 이상을 권장합니다.

```bash
cd vue
npm install
npm run dev
```

기본 개발 서버 주소는 다음과 같습니다.

```text
http://localhost:5173
```

## FastAPI 엔드포인트

| Method | Endpoint | 설명 |
| --- | --- | --- |
| GET | `/` | 서버 상태 확인 |
| GET | `/api/safety/all` | 지도 도로 오버레이용 전체 캐시 데이터 조회 |
| GET | `/api/safety/observations/{observation_id}` | DB 레코드 ID 기준 상세 안전도 조회 |
| GET | `/api/safety/{osmid}` | OSM ID 기준 상세 안전도 조회 |
| GET | `/predict?lat={lat}&lng={lng}&heading={heading}` | Street View 단일 방향 실시간 분석 |
| POST | `/predict-upload` | 업로드 이미지 실시간 분석 |

주요 응답 필드는 다음과 같습니다.

| 필드 | 설명 |
| --- | --- |
| `safety_score` | 1.0부터 5.0까지의 최종 안전도 |
| `analysis_basis` | 분석 방식: 캐시 구간 평균, 실시간 단일 방향, 업로드 이미지 |
| `explanation` | HTML 문자열 형태의 XAI 설명 |
| `image_url` | Street View 기반 분석일 때 사용한 이미지 URL |
| `features` | 모델 입력에 사용된 추출 특징 |

## 데이터베이스 스키마

`database/models.py`의 `SafetyObservation` 모델은 지도 서비스의 핵심 캐시 테이블입니다.

| 컬럼 | 설명 |
| --- | --- |
| `image_filename` | 분석 대상 또는 구간 식별자 |
| `osmid` | OpenStreetMap way ID |
| `segment_key` | OSM ID와 양 끝점 좌표로 만든 구간 고유 키 |
| `latitude`, `longitude` | 구간 중간점 |
| `start_latitude`, `start_longitude` | 구간 시작점 |
| `end_latitude`, `end_longitude` | 구간 끝점 |
| `source_dataset` | 데이터 출처 |
| `safety_score` | 최종 안전 점수 |
| `predicted_score` | 모델 예측 점수 |
| `model_name` | 사용 모델 설명 |
| `split` | train/test/full 등 데이터 분할명 |
| 특징 컬럼 | YOLO, SegFormer, OpenCV에서 추출한 모델 입력 특징 |

## 시연 시나리오

1. FastAPI 서버와 Vue 개발 서버를 실행합니다.
2. 브라우저에서 `http://localhost:5173`에 접속합니다.
3. 지도 서비스에서 `도보` 버튼을 눌러 성북구 도보 네트워크와 안전도 색상 레이어를 표시합니다.
4. 색상이 표시된 도로 구간을 클릭해 4방향 평균 기반 안전 점수와 설명을 확인합니다.
5. 도보 레이어를 끄고 지도 임의 위치를 클릭해 Street View 실시간 분석을 확인합니다.
6. `사진 분석` 메뉴에서 JPG 또는 PNG 거리 이미지를 업로드해 안전 점수와 설명을 확인합니다.

## 문제 해결

| 문제 | 원인 | 해결 |
| --- | --- | --- |
| `DATABASE_URL` 오류 | `.env`에 DB URL이 없음 | 프로젝트 루트 `.env`에 `DATABASE_URL=sqlite:///./mapsafe.db` 추가 |
| FastAPI 업로드 오류 | `python-multipart` 미설치 | `pip install python-multipart` 실행 |
| 지도 미표시 | Kakao JavaScript 키 누락 | `vue/.env`에 `VITE_KAKAO_MAP_KEY` 설정 |
| Street View 요청 실패 | Google API 키, 결제, 권한, 쿼터 문제 | `GOOGLE_API_KEY`와 Street View API 활성화 상태 확인 |
| Gemini 설명 미생성 | Gemini 키 없음 또는 쿼터 초과 | `GEMINI_API_KEY` 확인, 키가 없어도 로컬 설명으로 fallback |
| CUDA out of memory | GPU 메모리 부족 | CPU 실행 또는 입력 배치/이미지 수 조정 |
| SegFormer 다운로드 실패 | Hugging Face 모델 다운로드 문제 | 네트워크 확인 후 재실행, 캐시 디렉터리 설정 |
| LightGBM 오류 | macOS에서 OpenMP 런타임 누락 | `brew install libomp` 실행 |
| AutoGluon 설치 실패 | Python 버전 호환성 문제 | Python 3.10 또는 3.11 환경 사용 |

## 요약

MapSafe는 단순히 사진 한 장을 분류하는 모델이 아니라, 도로 구간 단위의 도시 안전 데이터를 생성하고 지도에서 바로 탐색할 수 있게 만든 서비스형 AI 시스템입니다. 사전 구축된 DB는 발표 현장에서 빠른 응답을 제공하고, 실시간 Street View 분석과 사진 업로드 분석은 모델이 새로운 입력에도 동작한다는 점을 보여줍니다.

핵심 차별점은 다음과 같습니다.

| 차별점 | 내용 |
| --- | --- |
| 멀티모달 비전 특징 | 객체 수, 도시 구성 비율, 밝기/엣지 특징을 함께 사용 |
| 구간 단위 캐싱 | 4방향 Street View 평균으로 도로 구간을 대표하는 안전도 산출 |
| 지도 기반 UX | 사용자가 안전/위험 구간을 직관적으로 탐색 가능 |
| 설명 가능한 결과 | SHAP 중요도와 Gemini 설명으로 예측 근거 제공 |
| 실시간 확장성 | 캐시된 구간뿐 아니라 임의 좌표와 업로드 이미지도 분석 가능 |

## 관리 참고

`.gitignore` 설정에 따라 데이터, 이미지, 모델 아티팩트, `.env` 파일은 기본적으로 Git에 포함되지 않습니다. 발표 또는 재현을 위해서는 다음 파일/폴더를 별도로 준비해야 합니다.

| 항목 | 용도 |
| --- | --- |
| `models/` | FastAPI 서버가 로드하는 AutoGluon 모델 |
| `models/shap_global_importance.csv` | Gemini 설명 생성 시 참고하는 전역 중요도 |
| `mapsafe.db` 또는 지정 DB | 지도 캐시 데이터 |
| `database/test_db.csv` | DB 적재용 사전 분석 CSV |
| `unique_coords_20m.csv` | 도로 구간 사전 분석 입력 |
| `images/`, `ground_truth.csv` | 모델 학습 입력 |
| `.env`, `vue/.env` | API 키와 DB 연결 정보 |

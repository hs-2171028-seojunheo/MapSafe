"""
데이터베이스 테이블(스키마) 정의 모듈
"""
from sqlalchemy import Column, Integer, Float, String, DateTime, Numeric
from sqlalchemy.sql import func
from .database_setup import Base

class SafetyObservation(Base):
    __tablename__ = "safety_observations"

    id = Column(Integer, primary_key=True, index=True)

    # 사진 소스 식별자 (어떤 구간의 사진을 분석했는지)
    image_filename = Column(String, nullable=False, unique=True, index=True)

    # 도로 전체 ID와 분석 구간 ID
    osmid = Column(String, nullable=True, index=True)
    segment_key = Column(String, nullable=True, unique=True, index=True)
    
    # 공간 데이터 (분석 구간의 중간점과 양 끝점)
    latitude = Column(Float, nullable=True, index=True)
    longitude = Column(Float, nullable=True, index=True)
    start_latitude = Column(Float, nullable=True)
    start_longitude = Column(Float, nullable=True)
    end_latitude = Column(Float, nullable=True)
    end_longitude = Column(Float, nullable=True)
    source_dataset = Column(String, nullable=False, default="unknown", index=True)

    # AI 예측 결과 (최종 안전도 점수)
    safety_score = Column(Float, nullable=False)

    # AI 추출 특징 (팝업창 및 SHAP 설명용 주요 위험/안전 요소)
    person_count = Column(Integer, default=0)
    car_count = Column(Integer, default=0)
    truck_count = Column(Integer, default=0)

    road_ratio = Column(Float, default=0.0)
    building_ratio = Column(Float, default=0.0)
    wall_ratio = Column(Float, default=0.0)
    vegetation_ratio = Column(Float, default=0.0)
    sky_ratio = Column(Float, default=0.0)

    brightness_mean = Column(Float, nullable=True)
    dark_area_ratio = Column(Float, nullable=True)
    edge_density = Column(Float, nullable=True)

    model_name = Column(String, nullable=True)
    predicted_score = Column(Float, nullable=True)
    split = Column(String, nullable=True, index=True)

    # 레코드 생성/수정 시간
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return (
            "<SafetyObservation("
            f"id={self.id}, image={self.image_filename}, score={self.safety_score}, "
            f"source={self.source_dataset})>"
        )

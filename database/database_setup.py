"""
데이터베이스 연결 및 엔진 설정 모듈
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

if not SQLALCHEMY_DATABASE_URL:
    raise ValueError("DATABASE_URL이 없습니다! .env 파일을 확인해주세요.")

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """FastAPI 등에서 DB 세션을 안전하게 열고 닫기 위한 의존성 함수"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
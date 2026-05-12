import pandas as pd
import numpy as np

def preprocess_survey_data(raw_csv_path: str, output_csv_path: str) -> None:
    # 1. 설문 원본 데이터 로드
    print("[System] 원본 설문 데이터 로딩 중...")
    df = pd.read_excel(raw_csv_path)
    
    records = []
    
    # 2. 모든 응답자(Row)를 순회하며 데이터 쪼개기 (Wide to Long)
    for index, row in df.iterrows():
        # 1쌍부터 25쌍까지 반복
        for i in range(1, 26):
            # A사진 추출 (Q2가 A사진의 안전도 점수)
            img_a_id = row.get(f'쌍{i}_A사진')
            score_a = row.get(f'쌍{i}_Q2')
            if pd.notna(img_a_id) and pd.notna(score_a):
                records.append({'image_id': img_a_id, 'safety_score': score_a})
                
            # B사진 추출 (Q3가 B사진의 안전도 점수)
            img_b_id = row.get(f'쌍{i}_B사진')
            score_b = row.get(f'쌍{i}_Q3')
            if pd.notna(img_b_id) and pd.notna(score_b):
                records.append({'image_id': img_b_id, 'safety_score': score_b})

    # 3. 세로로 긴 데이터프레임 생성
    long_df = pd.DataFrame(records)
    
    # 예외 처리: 숫자로 변환 안 되는 값(결측치 등) 강제 변환 후 제거
    long_df['safety_score'] = pd.to_numeric(long_df['safety_score'], errors='coerce')
    long_df = long_df.dropna()
    
    # 4. 동일한 사진 ID에 대해 점수 평균 내기
    final_df = long_df.groupby('image_id')['safety_score'].mean().reset_index()
    
    # 5. image_filename 형태로 변환 (예: 135 -> 135.jpg)
    # 실제 파일 확장자에 맞게 수정
    final_df['image_filename'] = final_df['image_id'].astype(str) + '.jpg'
    
    # AutoGluon이 요구하는 2개 컬럼만 남기기
    final_df = final_df[['image_filename', 'safety_score']]
    
    # 6. 최종 파일 저장
    final_df.to_csv(output_csv_path, index=False)
    
    print(f"[System] 전처리 완료! 총 {len(final_df)}장의 고유 이미지 정답지 생성됨.")
    print(f"[System] 출력 파일: {output_csv_path}")
    print("\n--- [데이터 샘플 (상위 5개)] ---")
    print(final_df.head())

if __name__ == "__main__":
    # 엑셀(CSV) 파일명과 일치
    preprocess_survey_data(
        raw_csv_path="./preprocess/도시 안전 인식 조사 결과.xlsx", 
        output_csv_path="./ground_truth.csv"
    )

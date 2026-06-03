"""
AutoGluon model module for urban safety score prediction.
This module trains, evaluates, and predicts from merged feature data.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import psutil
import torch
from autogluon.tabular import TabularPredictor

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SafetyModelPredictor:
    """Train and use an AutoGluon regressor for safety score prediction."""

    def __init__(
        self,
        extracted_features_path: str = "./extracted_features.csv",
        ground_truth_path: str = "./ground_truth.csv",
        model_dir: str = "./models/",
    ) -> None:
        """Initialize model predictor paths.

        Args:
            extracted_features_path: Path to merged extracted feature CSV.
            ground_truth_path: Path to ground truth CSV.
            model_dir: Output path for AutoGluon artifacts.
        """
        self.extracted_features_path = Path(extracted_features_path)
        self.ground_truth_path = Path(ground_truth_path)
        self.model_dir = model_dir
        self.predictor: Optional[TabularPredictor] = None

    def _validate_input_files(self) -> None:
        """Validate required input files for model training."""
        missing_files = []
        if not self.extracted_features_path.exists():
            missing_files.append(str(self.extracted_features_path))
        if not self.ground_truth_path.exists():
            missing_files.append(str(self.ground_truth_path))

        if missing_files:
            raise FileNotFoundError(
                "Required training files are missing: " + ", ".join(missing_files)
            )

    def prepare_training_data(self) -> pd.DataFrame:
        """Merge extracted features with ground truth and prepare training data.

        Returns:
            DataFrame containing features and label for training.
        """
        self._validate_input_files()

        features_df = pd.read_csv(self.extracted_features_path)
        ground_truth_df = pd.read_csv(self.ground_truth_path)

        required_ground_truth_columns = {"image_filename", "safety_score"}
        if not required_ground_truth_columns.issubset(set(ground_truth_df.columns)):
            raise ValueError(
                "ground_truth.csv must include columns: image_filename, safety_score"
            )
        if "image_filename" not in features_df.columns:
            raise ValueError("extracted_features.csv must include 'image_filename' column.")

        merged_df = features_df.merge(ground_truth_df, on="image_filename", how="inner")
        if merged_df.empty:
            raise ValueError(
                "Merged training data is empty. Check filename consistency between files."
            )

        columns_to_drop = ["image_filename", "latitude", "longitude", "lat", "lng"] 
        
        existing_cols = [col for col in columns_to_drop if col in merged_df.columns]
        training_df = merged_df.drop(columns=existing_cols)
        
        return training_df

    def train(self, training_data: pd.DataFrame, time_limit: int = 300) -> TabularPredictor:
        """Train AutoGluon TabularPredictor.

        Args:
            training_data: DataFrame with features and safety_score label.
            time_limit: AutoGluon training time budget in seconds.

        Returns:
            Trained TabularPredictor.
        """
        if "safety_score" not in training_data.columns:
            raise ValueError("Training data must include 'safety_score' label column.")

        self.predictor = TabularPredictor(
            label="safety_score",
            problem_type="regression",
            path=self.model_dir,
        )
        self.predictor.fit(train_data=training_data, time_limit=time_limit, verbosity=1)

        print("[Model] Training complete.")
        print(f"[Model] Model artifacts saved to: {self.model_dir}")
        return self.predictor

    def evaluate(self, data_with_label: pd.DataFrame) -> Dict:
        """Evaluate trained model.

        Args:
            data_with_label: DataFrame including safety_score label.

        Returns:
            Dictionary with evaluation metrics.
        """
        if self.predictor is None:
            raise RuntimeError("Model is not trained yet.")
        if "safety_score" not in data_with_label.columns:
            raise ValueError("Evaluation data must include 'safety_score' label column.")

        metrics = self.predictor.evaluate(data_with_label)
        print(f"[Model] Evaluation metrics: {metrics}")
        return metrics

    def predict(self, features_only: pd.DataFrame) -> pd.Series:
        """Predict safety score from feature-only DataFrame.

        Args:
            features_only: DataFrame with feature columns only.

        Returns:
            Predicted safety score series.
        """
        if self.predictor is None:
            raise RuntimeError("Model is not trained yet.")

        return self.predictor.predict(features_only)
    
    def analyze_shap(self, training_data: pd.DataFrame, sample_size: int = 50) -> None:
        """
        SHAP 분석 결과를 이미지뿐만 아니라 수치 데이터(CSV)로 추출하여 저장합니다.
        """
        if self.predictor is None:
            raise RuntimeError("모델이 먼저 학습되어야 합니다.")

        print("\n[SHAP] SHAP 분석을 준비합니다...")
        
        import numpy as np
        
        feature_cols = self.predictor.feature_metadata_in.get_features()
        features_only = training_data[feature_cols]

        background_data = features_only.sample(n=min(100, len(features_only)), random_state=42)
        sample_data = features_only.sample(n=min(sample_size, len(features_only)), random_state=42)

        def predict_wrapper(x_array):
            df = pd.DataFrame(x_array, columns=feature_cols)
            return self.predictor.predict(df).values

        print("[SHAP] KernelExplainer 초기화 중...")
        explainer = shap.KernelExplainer(predict_wrapper, background_data)
        
        print(f"[SHAP] {len(sample_data)}개 샘플에 대한 SHAP Value 계산 중...")
        # ncores=1 옵션을 유지하여 macOS Segfault 방지
        shap_values = explainer.shap_values(sample_data, ncores=1) 

        # 다중 출력(리스트)인 경우 단일 배열로 변환
        if isinstance(shap_values, list):
            shap_values_array = shap_values[0]
        else:
            shap_values_array = shap_values

        # ---------------------------------------------------------
        # 1. 평균 특성 중요도 (Global Importance) CSV 추출
        # 각 특성의 SHAP 절댓값 평균을 구하여 어떤 변수가 가장 중요한지 랭킹 매김
        # ---------------------------------------------------------
        mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
        
        global_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Mean_Absolute_SHAP': mean_abs_shap
        }).sort_values(by='Mean_Absolute_SHAP', ascending=False)
        
        global_csv_path = Path(self.model_dir) / "shap_global_importance.csv"
        global_importance_df.to_csv(global_csv_path, index=False)
        print(f"[SHAP] 전체 특성 중요도 데이터가 저장되었습니다: {global_csv_path}")

        # ---------------------------------------------------------
        # 2. 개별 샘플의 SHAP 값 (Local Importance) CSV 추출
        # 각 사진(샘플)마다 변수들이 점수에 + / - 를 얼마나 줬는지 기록
        # ---------------------------------------------------------
        local_shap_df = pd.DataFrame(shap_values_array, columns=feature_cols)
        local_csv_path = Path(self.model_dir) / "shap_local_values.csv"
        local_shap_df.to_csv(local_csv_path, index=False)
        print(f"[SHAP] 개별 샘플의 상세 분석 데이터가 저장되었습니다: {local_csv_path}")
        
        # (선택) 기존 이미지 저장 코드 유지
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_array, sample_data, show=False)
        plt.tight_layout()
        
        plot_path = Path(self.model_dir) / "shap_summary.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()


def log_memory_profile() -> None:
    """Log process RAM and CUDA VRAM usage in MB."""
    process = psutil.Process()
    peak_ram_mb = process.memory_info().rss / (1024 * 1024)

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(
        f"[Memory Profiler] Peak RAM: {peak_ram_mb:.0f} MB | "
        f"Peak VRAM: {peak_vram_mb:.0f} MB"
    )


def main() -> None:
    """Entrypoint for model training module."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model_predictor = SafetyModelPredictor(
        extracted_features_path="./extracted_features.csv",
        ground_truth_path="./ground_truth.csv",
        model_dir="./models/",
    )

    try:
        training_df = model_predictor.prepare_training_data()
        predictor = model_predictor.train(training_data=training_df, time_limit=300)
        
        # Demonstration-only evaluation on same training data.
        model_predictor.evaluate(training_df)

        # SHAP 분석 추가 실행
        model_predictor.analyze_shap(training_df, sample_size=50)

        sample_features = training_df.drop(columns=["safety_score"]).head(5)
        if not sample_features.empty:
            sample_predictions = model_predictor.predict(sample_features)
            print("[Model] Sample predictions (first 5 rows):")
            for index, pred in enumerate(sample_predictions, start=1):
                print(f"  Sample {index}: {pred:.4f}")
    except Exception as exc:
        print(f"[Model][ERROR] {exc}")
    finally:
        log_memory_profile()

if __name__ == "__main__":
    main()
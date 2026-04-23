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

        # Prevent data leakage by dropping image identifier before training.
        training_df = merged_df.drop(columns=["image_filename"])
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
        _ = predictor

        # Demonstration-only evaluation on same training data.
        model_predictor.evaluate(training_df)

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

"""
Test script for the split pipeline architecture.
It stages images from test folders, runs extraction in isolated subprocesses,
merges CSV outputs, and trains/evaluates a test model.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import FeatureMerger


def collect_images_to_staging(source_paths: list[str], staging_dir: Path) -> int:
    """
    Copy images from source folders into a single staging directory.

    Args:
        source_paths: List of source image folders.
        staging_dir: Destination directory for staged images.

    Returns:
        Number of staged images.
    """
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged_count = 0

    for idx, image_dir in enumerate(source_paths, 1):
        folder = Path(image_dir)

        if not folder.exists():
            print(f"[Warning] Missing folder: {folder}")
            continue

        image_files = sorted(
            list(folder.glob("*.jpg"))
            + list(folder.glob("*.jpeg"))
            + list(folder.glob("*.png"))
        )

        print(f"[{idx}/{len(source_paths)}] Staging from: {folder.name} ({len(image_files)} files)")

        for image_file in image_files:
            target_name = f"{folder.name}_{image_file.name}"
            shutil.copy2(image_file, staging_dir / target_name)
            staged_count += 1

    return staged_count


def run_python_subprocess(code_snippet: str, stage_name: str) -> None:
    """Run a Python code snippet in an isolated subprocess.

    Args:
        code_snippet: Python code to execute.
        stage_name: Display name for logging.
    """
    print(f"[Subprocess] {stage_name}")
    subprocess.run([sys.executable, "-c", code_snippet], check=True)


def run_split_pipeline_test(source_paths: list[str]) -> None:
    """Run full test flow using split architecture modules.

    Args:
        source_paths: List of source image folders.
    """
    print("=" * 70)
    print("Split Pipeline Test Runner")
    print("=" * 70)

    staging_dir = Path("./images_test_staging")
    staged_count = collect_images_to_staging(source_paths, staging_dir)
    if staged_count == 0:
        print("[Error] No images staged. Aborting test.")
        return

    print(f"[Info] Total staged images: {staged_count}")

    yolo_csv = Path("./yolo_features_test.csv")
    segformer_csv = Path("./segformer_features_test.csv")
    extracted_csv = Path("./extracted_features_test.csv")
    ground_truth_csv = Path("./ground_truth_test.csv")
    model_dir = Path("./models_test/")

    try:
        # Stage 1: YOLO extraction (isolated process)
        run_python_subprocess(
            (
                "from extractors.extractor_yolo import YoloFeatureExtractor; "
                f"YoloFeatureExtractor('yolo11n.pt').run(image_dir={str(staging_dir)!r}, "
                f"output_csv={str(yolo_csv)!r})"
            ),
            "Stage 1 - YOLO Extraction",
        )

        # Stage 2: SegFormer extraction (isolated process)
        run_python_subprocess(
            (
                "from extractors.extractor_segformer import SegFormerFeatureExtractor; "
                f"SegFormerFeatureExtractor(device=None).run(image_dir={str(staging_dir)!r}, "
                f"output_csv={str(segformer_csv)!r})"
            ),
            "Stage 2 - SegFormer Extraction",
        )

        # Stage 3: Merge extraction outputs
        print("[Stage 3] Merge CSV files")
        merger = FeatureMerger(
            yolo_csv_path=str(yolo_csv),
            segformer_csv_path=str(segformer_csv),
            output_csv_path=str(extracted_csv),
        )
        merger.run()

        combined_features = pd.read_csv(extracted_csv)
        print("[Info] Extracted feature preview:")
        print(combined_features.head())

        # Stage 4: Create synthetic test ground truth
        print("[Stage 4] Create synthetic ground truth")
        safety_scores = np.round(np.random.uniform(2.5, 4.5, len(combined_features)), 1)
        ground_truth = pd.DataFrame(
            {
                "image_filename": combined_features["image_filename"],
                "safety_score": safety_scores,
            }
        )
        ground_truth.to_csv(ground_truth_csv, index=False, float_format="%.1f")
        print(f"[Info] Ground truth saved: {ground_truth_csv}")

        # Stage 5: Training and evaluation (isolated process)
        run_python_subprocess(
            (
                "import pandas as pd; "
                "from model_predictor import SafetyModelPredictor; "
                f"mp=SafetyModelPredictor(extracted_features_path={str(extracted_csv)!r}, "
                f"ground_truth_path={str(ground_truth_csv)!r}, model_dir={str(model_dir)!r}); "
                "td=mp.prepare_training_data(); "
                "mp.train(training_data=td, time_limit=300); "
                "mp.evaluate(td); "
                f"merged=pd.read_csv({str(extracted_csv)!r}).merge(pd.read_csv({str(ground_truth_csv)!r}), on='image_filename', how='inner'); "
                "sample=merged.head(10).copy(); "
                "sample_features=sample.drop(columns=['image_filename','safety_score']); "
                "pred=mp.predict(sample_features); "
                "print('[Model] Sample predictions with filenames:'); "
                    "lines=[f'  {fname} | Predicted: {pred_score:.1f} | Actual: {actual:.1f}' "
                    "for fname,pred_score,actual in zip(sample['image_filename'], pred, sample['safety_score'])]; "
                    "print('\\n'.join(lines))"
            ),
            "Stage 5 - AutoGluon Train/Evaluate",
        )

        print("=" * 70)
        print("Test completed successfully.")
        print("Generated artifacts:")
        print(f"  - {yolo_csv}")
        print(f"  - {segformer_csv}")
        print(f"  - {extracted_csv}")
        print(f"  - {ground_truth_csv}")
        print(f"  - {model_dir}")
        print("=" * 70)
    except subprocess.CalledProcessError as exc:
        print(f"[Error] Subprocess stage failed with code {exc.returncode}: {exc}")
    except Exception as exc:
        print(f"[Error] Test pipeline failed: {exc}")
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def main() -> None:
    """Main function."""
    # Define source image folders for testing.
    image_folders = [
        r'C:\Users\ICT\MapSafe\test_images\건물 유무(70장)\건물O(50장)',
        r'C:\Users\ICT\MapSafe\test_images\사람(50장)',
        r'C:\Users\ICT\MapSafe\test_images\도로(50장)',
    ]

    print("\n[Config] Source folders:")
    for folder in image_folders:
        folder_path = Path(folder)
        if folder_path.exists():
            image_count = len(list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png')))
            status = "OK" if image_count > 0 else "EMPTY"
            print(f"  [{status}] {folder_path.name:30s} ({image_count} images)")
        else:
            print(f"  [MISSING] {folder_path.name:30s} (folder not found)")
    print()

    run_split_pipeline_test(image_folders)

    print("\n[Tip]")
    print("  - extracted_features_test.csv: merged test features")
    print("  - ground_truth_test.csv: synthetic labels for test run")
    print("  - models_test/: trained model artifacts")
    print()


if __name__ == "__main__":
    main()

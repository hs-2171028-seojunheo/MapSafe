"""
YOLO feature extractor module.
This module extracts dynamic object counts from images and saves them to CSV.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
import psutil
import torch
from ultralytics import YOLO


class YoloFeatureExtractor:
    """Extract dynamic object count features using YOLO11n."""

    def __init__(self, model_path: str = "yolo11n.pt") -> None:
        """Initialize the YOLO feature extractor.

        Args:
            model_path: Path to YOLO model weights.
        """
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.target_classes: Dict[str, int] = {
            "person": 0,
            "car": 2,
            "truck": 7,
        }

    def _extract_single_image(self, image_path: Path) -> Dict[str, int]:
        """Extract YOLO object counts from one image.

        Args:
            image_path: Image file path.

        Returns:
            Dictionary of object counts.
        """
        counts: Dict[str, int] = {
            "person_count": 0,
            "car_count": 0,
            "truck_count": 0,
        }

        results = self.model(str(image_path), verbose=False)
        for result in results:
            if result.boxes is None:
                continue

            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for class_id in class_ids:
                if class_id == self.target_classes["person"]:
                    counts["person_count"] += 1
                elif class_id == self.target_classes["car"]:
                    counts["car_count"] += 1
                elif class_id == self.target_classes["truck"]:
                    counts["truck_count"] += 1

        return counts

    def extract_from_directory(self, image_dir: str = "./images/") -> pd.DataFrame:
        """Extract YOLO features from all images in a directory.

        Args:
            image_dir: Directory that contains image files.

        Returns:
            DataFrame with YOLO features.
        """
        image_path = Path(image_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        image_files: List[Path] = sorted(
            list(image_path.glob("*.jpg"))
            + list(image_path.glob("*.jpeg"))
            + list(image_path.glob("*.png"))
        )
        if not image_files:
            raise ValueError(f"No image files found in directory: {image_dir}")

        rows: List[Dict[str, object]] = []
        for file_path in image_files:
            print(f"[YOLO] Processing: {file_path.name}")
            try:
                row: Dict[str, object] = {
                    "image_filename": file_path.name,
                    **self._extract_single_image(file_path),
                }
                rows.append(row)
            except Exception as exc:
                print(f"[YOLO][ERROR] Failed to process {file_path.name}: {exc}")

        if not rows:
            raise RuntimeError("YOLO extraction completed with zero successful images.")

        return pd.DataFrame(rows)

    def run(self, image_dir: str = "./images/", output_csv: str = "./yolo_features.csv") -> None:
        """Run full YOLO extraction and write CSV output.

        Args:
            image_dir: Input image directory.
            output_csv: Destination CSV path.
        """
        try:
            df = self.extract_from_directory(image_dir=image_dir)
            df.to_csv(output_csv, index=False)
            print(f"[YOLO] Feature extraction complete: {output_csv}")
            print(f"[YOLO] Extracted rows: {len(df)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[YOLO] CUDA cache cleared.")


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
    """Entrypoint for YOLO extractor module."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    extractor = YoloFeatureExtractor(model_path="yolo11n.pt")
    try:
        extractor.run(image_dir="./images/", output_csv="./yolo_features.csv")
    finally:
        log_memory_profile()


if __name__ == "__main__":
    main()

"""
OpenCV feature extractor module.
This module extracts handcrafted visual features from images and saves them to CSV.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import psutil
import torch


class OpenCVFeatureExtractor:
    """Extract brightness, darkness, edge, road width, sidewalk proxy features."""
    def __init__(self, device=None):

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[OpenCV] Device: {self.device}")

    @staticmethod
    def _read_image(image_path: Path) -> Optional[Any]:
        image = cv2.imread(str(image_path))
        if image is not None:
            return image

        try:
            file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
            if file_bytes.size == 0:
                return None
            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _extract_single_image(self, image_path: Path) -> Dict[str, float]:
        image = self._read_image(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 전체 밝기 평균
        brightness_mean = float(np.mean(gray))

        # 2. 어두운 영역 비율
        dark_threshold = 60
        dark_area_ratio = float(np.sum(gray < dark_threshold) / gray.size * 100)

        # 3. 엣지 밀도
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        edge_density = float(np.sum(edges > 0) / edges.size * 100)

        return {
            "brightness_mean": round(brightness_mean, 2),
            "dark_area_ratio": round(dark_area_ratio, 2),
            "edge_density": round(edge_density, 2),
        }

    def extract_from_directory(self, image_dir: str = "./images/new1") -> pd.DataFrame:
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
            print(f"[OpenCV] Processing: {file_path.name}")

            try:
                row: Dict[str, object] = {
                    "image_filename": file_path.name,
                    **self._extract_single_image(file_path),
                }
                rows.append(row)
            except Exception as exc:
                print(f"[OpenCV][ERROR] Failed to process {file_path.name}: {exc}")

        if not rows:
            raise RuntimeError("OpenCV extraction completed with zero successful images.")

        return pd.DataFrame(rows)

    def run(
        self,
        image_dir: str = "./images/",
        output_csv: str = "./opencv_features.csv",
    ) -> None:
        try:
            df = self.extract_from_directory(image_dir=image_dir)
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"[OpenCV] Feature extraction complete: {output_csv}")
            print(f"[OpenCV] Extracted rows: {len(df)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[OpenCV] CUDA cache cleared.")


def log_memory_profile() -> None:
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
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    extractor = OpenCVFeatureExtractor()

    try:
        extractor.run(
            image_dir="./images/new1",
            output_csv="./opencv_features1.csv",
        )
    finally:
        log_memory_profile()


if __name__ == "__main__":
    main()
"""
SegFormer feature extractor module.
This module extracts semantic segmentation ratios from images and saves them to CSV.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import psutil
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


class SegFormerFeatureExtractor:
    """Extract semantic pixel ratio features using SegFormer."""

    def __init__(self, device: Optional[str] = None) -> None:
        """Initialize SegFormer feature extractor.

        Args:
            device: Device string. If None, auto-detects CUDA or CPU.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        model_name = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(self.device)

        self.target_classes: Dict[str, int] = {
            "road": 0,
            "building": 2,
            "wall": 3,
            "vegetation": 8,
            "sky": 10,
        }

    @staticmethod
    def _read_image(image_path: Path) -> Optional[Any]:
        """Read image robustly for Unicode paths on Windows.

        Args:
            image_path: Image file path.

        Returns:
            Decoded BGR image, or None when reading fails.
        """
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
        """Extract segmentation class ratios from one image.

        Args:
            image_path: Image file path.

        Returns:
            Dictionary with class area ratios in percent.
        """
        image = self._read_image(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_mask = torch.argmax(outputs.logits.cpu(), dim=1)[0]
        total_pixels = int(predicted_mask.shape[0] * predicted_mask.shape[1])

        ratios: Dict[str, float] = {
            "road_ratio": 0.0,
            "building_ratio": 0.0,
            "wall_ratio": 0.0,
            "vegetation_ratio": 0.0,
            "sky_ratio": 0.0,
        }

        for class_name, class_id in self.target_classes.items():
            pixel_count = int((predicted_mask == class_id).sum().item())
            ratio = round((pixel_count / total_pixels) * 100.0, 2)
            ratios[f"{class_name}_ratio"] = ratio

        return ratios

    def extract_from_directory(self, image_dir: str = "./images/") -> pd.DataFrame:
        """Extract SegFormer features from all images in a directory.

        Args:
            image_dir: Directory that contains image files.

        Returns:
            DataFrame with segmentation ratio features.
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
            print(f"[SegFormer] Processing: {file_path.name}")
            try:
                row: Dict[str, object] = {
                    "image_filename": file_path.name,
                    **self._extract_single_image(file_path),
                }
                rows.append(row)
            except Exception as exc:
                print(f"[SegFormer][ERROR] Failed to process {file_path.name}: {exc}")

        if not rows:
            raise RuntimeError("SegFormer extraction completed with zero successful images.")

        return pd.DataFrame(rows)

    def run(
        self,
        image_dir: str = "./images/",
        output_csv: str = "./segformer_features.csv",
    ) -> None:
        """Run full SegFormer extraction and write CSV output.

        Args:
            image_dir: Input image directory.
            output_csv: Destination CSV path.
        """
        try:
            df = self.extract_from_directory(image_dir=image_dir)
            df.to_csv(output_csv, index=False)
            print(f"[SegFormer] Feature extraction complete: {output_csv}")
            print(f"[SegFormer] Extracted rows: {len(df)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[SegFormer] CUDA cache cleared.")


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
    """Entrypoint for SegFormer extractor module."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    extractor = SegFormerFeatureExtractor(device=None)
    try:
        extractor.run(image_dir="./images/", output_csv="./segformer_features.csv")
    finally:
        log_memory_profile()


if __name__ == "__main__":
    main()

"""
데이터베이스 적재 모듈
"""
import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import or_

from database.database_setup import Base, SessionLocal, engine
from database.models import SafetyObservation


MODEL_COLUMNS = {
    "image_filename",
    "osmid",
    "segment_key",
    "latitude",
    "longitude",
    "start_latitude",
    "start_longitude",
    "end_latitude",
    "end_longitude",
    "source_dataset",
    "safety_score",
    "person_count",
    "car_count",
    "truck_count",
    "road_ratio",
    "building_ratio",
    "wall_ratio",
    "vegetation_ratio",
    "sky_ratio",
    "brightness_mean",
    "dark_area_ratio",
    "edge_density",
    "model_name",
    "predicted_score",
    "split",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Import MapSafe feature CSV into SQLite.")
    parser.add_argument("--csv", required=True, help="Feature CSV to import.")
    parser.add_argument("--source", default="manual", help="Dataset/source name.")
    parser.add_argument("--split", default=None, help="Optional train/test/full split label.")
    parser.add_argument("--replace", action="store_true", help="Replace rows with same image_filename.")
    return parser.parse_args()


def normalize_columns(df, source, split):
    rename_map = {}
    if "filename" in df.columns and "image_filename" not in df.columns:
        rename_map["filename"] = "image_filename"
    if "lat" in df.columns and "latitude" not in df.columns:
        rename_map["lat"] = "latitude"
    if "lng" in df.columns and "longitude" not in df.columns:
        rename_map["lng"] = "longitude"
    df = df.rename(columns=rename_map)

    if "image_filename" not in df.columns:
        raise ValueError("CSV must include image_filename or filename.")
    if "safety_score" not in df.columns:
        raise ValueError("CSV must include safety_score.")

    df["source_dataset"] = source
    if split is not None:
        df["split"] = split

    return df


def row_to_observation(row):
    payload = {}
    for col in MODEL_COLUMNS:
        if col in row and pd.notna(row[col]):
            value = row[col]
            if hasattr(value, "item"):
                value = value.item()
            payload[col] = value
    return SafetyObservation(**payload)


def import_csv(csv_path, source, split, replace):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    Base.metadata.create_all(bind=engine)
    df = normalize_columns(pd.read_csv(path), source=source, split=split)

    db = SessionLocal()
    try:
        imported = 0
        for _, row in df.iterrows():
            if replace:
                conditions = [SafetyObservation.image_filename == row["image_filename"]]
                if "segment_key" in row and pd.notna(row["segment_key"]):
                    conditions.append(SafetyObservation.segment_key == row["segment_key"])
                existing = (
                    db.query(SafetyObservation)
                    .filter(or_(*conditions))
                    .first()
                )
                if existing is not None:
                    db.delete(existing)
                    db.flush()

            db.add(row_to_observation(row))
            imported += 1

        db.commit()
        print(f"Imported {imported} rows from {path}")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main():
    args = parse_args()
    import_csv(args.csv, source=args.source, split=args.split, replace=args.replace)


if __name__ == "__main__":
    main()

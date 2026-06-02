"""
.0 정규화 / 시작점·끝점 기반 segment_key 생성 모듈
"""
import hashlib
import re


OSMID_PREFIX = "osmid_"
INTEGER_FLOAT_PATTERN = re.compile(r"^(\d+)\.0+$")


def normalize_osmid(value) -> str:
    """Return a stable string representation for an OpenStreetMap way ID."""
    text = str(value).strip()
    if text.startswith(OSMID_PREFIX):
        text = text[len(OSMID_PREFIX):]

    match = INTEGER_FLOAT_PATTERN.fullmatch(text)
    return match.group(1) if match else text


def osmid_from_image_filename(image_filename: str) -> str:
    return normalize_osmid(image_filename or "")


def osmid_image_filename_candidates(osmid) -> list[str]:
    normalized = normalize_osmid(osmid)
    filenames = [f"{OSMID_PREFIX}{normalized}"]

    if normalized.isdigit():
        filenames.append(f"{OSMID_PREFIX}{normalized}.0")

    return filenames


def build_segment_key(osmid, start_latitude, start_longitude, end_latitude, end_longitude) -> str:
    normalized_osmid = normalize_osmid(osmid)
    endpoints = sorted([
        (float(start_latitude), float(start_longitude)),
        (float(end_latitude), float(end_longitude)),
    ])
    coordinate_text = "|".join(
        f"{latitude:.7f},{longitude:.7f}"
        for latitude, longitude in endpoints
    )
    digest = hashlib.sha256(coordinate_text.encode("ascii")).hexdigest()[:16]
    return f"{normalized_osmid}_{digest}"

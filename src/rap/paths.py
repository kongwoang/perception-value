"""Canonical paths. Everything else imports from here so no script hardcodes a path."""
from pathlib import Path
import os

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("RAP_DATA", "/home/kongwoang/datasets/kitti_tracking"))
MODELS = Path(os.environ.get("RAP_MODELS", "/home/kongwoang/models/yolo"))

KITTI_IMAGES = DATA / "training" / "image_02"
KITTI_LABELS = DATA / "training" / "label_02"
KITTI_CALIB = DATA / "training" / "calib"
KITTI_OXTS = DATA / "training" / "oxts"

CACHE = REPO / "data" / "cache"
RESULTS = REPO / "results"
RAW = RESULTS / "raw"
PROCESSED = RESULTS / "processed"
FIGURES = RESULTS / "figures"
DOCS = REPO / "docs"

for _p in (CACHE, RAW, PROCESSED, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

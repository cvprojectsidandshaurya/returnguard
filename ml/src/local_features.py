"""Local-detail matching for garment tags, logos, prints, and embroidery.

Global embeddings are strong at product identity but often miss small physical
cues. This classical ORB plus RANSAC baseline is cheap enough to run alongside
the embedding model and produces interpretable evidence: feature matches and
geometrically consistent inliers. It is evidence only, never a fraud verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class LocalMatchScore:
    reference_keypoints: int
    query_keypoints: int
    ratio_test_matches: int
    geometric_inliers: int
    inlier_ratio: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BestLocalMatch:
    packing_index: int
    rider_index: int
    score: LocalMatchScore

    def as_dict(self) -> dict:
        result = asdict(self)
        result["score"] = self.score.as_dict()
        return result


def _read_gray(path: Path, max_side: int = 1600) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    height, width = image.shape
    scale = max_side / max(height, width)
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def _empty_score(reference_keypoints: int, query_keypoints: int) -> LocalMatchScore:
    return LocalMatchScore(reference_keypoints, query_keypoints, 0, 0, 0.0)


def score_local_features(reference: np.ndarray, query: np.ndarray, ratio_threshold: float = 0.75) -> LocalMatchScore:
    """Match two grayscale images with ORB, a ratio test, and RANSAC homography."""
    if not 0.0 < ratio_threshold < 1.0:
        raise ValueError("ratio_threshold must be between zero and one")
    if reference.ndim != 2 or query.ndim != 2:
        raise ValueError("reference and query must be grayscale images")

    detector = cv2.ORB_create(nfeatures=2_000, fastThreshold=10)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(reference, None)
    query_keypoints, query_descriptors = detector.detectAndCompute(query, None)
    if reference_descriptors is None or query_descriptors is None:
        return _empty_score(len(reference_keypoints), len(query_keypoints))

    raw_matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(query_descriptors, reference_descriptors, k=2)
    good = [first for pair in raw_matches if len(pair) == 2 for first, second in [pair] if first.distance < ratio_threshold * second.distance]
    if len(good) < 4:
        return LocalMatchScore(len(reference_keypoints), len(query_keypoints), len(good), 0, 0.0)

    query_points = np.float32([query_keypoints[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    reference_points = np.float32([reference_keypoints[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(query_points, reference_points, cv2.RANSAC, ransacReprojThreshold=5.0)
    inliers = int(mask.ravel().sum()) if mask is not None else 0
    return LocalMatchScore(
        reference_keypoints=len(reference_keypoints),
        query_keypoints=len(query_keypoints),
        ratio_test_matches=len(good),
        geometric_inliers=inliers,
        inlier_ratio=float(inliers / len(good)),
    )


def score_local_paths(reference: Path | str, query: Path | str, ratio_threshold: float = 0.75) -> LocalMatchScore:
    return score_local_features(_read_gray(Path(reference)), _read_gray(Path(query)), ratio_threshold)


def best_local_match(packing_paths: list[Path], rider_paths: list[Path], ratio_threshold: float = 0.75) -> BestLocalMatch:
    """Return the strongest geometric local match across two image sets."""
    if not packing_paths or not rider_paths:
        raise ValueError("packing_paths and rider_paths must both be non-empty")
    best: BestLocalMatch | None = None
    for packing_index, packing in enumerate(packing_paths):
        for rider_index, rider in enumerate(rider_paths):
            score = score_local_paths(packing, rider, ratio_threshold)
            candidate = BestLocalMatch(packing_index, rider_index, score)
            if best is None or (score.geometric_inliers, score.inlier_ratio) > (best.score.geometric_inliers, best.score.inlier_ratio):
                best = candidate
    assert best is not None
    return best

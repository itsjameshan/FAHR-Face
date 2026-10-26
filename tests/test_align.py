"""The alignment port must be bit-identical to retina-face 0.0.17.

The oracle below is vendored *verbatim* from
``retinaface/commons/postprocess.py`` (retina-face 0.0.17) plus a literal
re-statement of the ``align=True`` branch of ``RetinaFace.extract_faces``.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import numpy as np
import pytest
from PIL import Image

from app.align import align_and_crop

# --------------------------------------------------------------------------- #
# Oracle: retina-face 0.0.17, retinaface/commons/postprocess.py (verbatim)
# --------------------------------------------------------------------------- #


def find_euclidean_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> float:
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def alignment_procedure(
    img: np.ndarray, left_eye: tuple, right_eye: tuple, nose: tuple
) -> Tuple[np.ndarray, float, int]:
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = find_euclidean_distance(np.array(right_eye), np.array(left_eye))

    # -----------------------
    # apply cosine rule
    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)

        # PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0],
        # floating point errors would produce cases violating this
        # In fact, we did come across a case where cos_a took the value 1.0000000169176173
        # which lead to a NaN from the following np.arccos step
        cos_a = min(1.0, max(-1.0, cos_a))

        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
    else:
        angle = 0.0  # Dummy value for undefined angle

    # -----------------------

    return img, angle, direction


def rotate_facial_area(
    facial_area: Tuple[int, int, int, int], angle: float, direction: int, size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    # Angle in radians
    angle = angle * np.pi / 180

    height, weight = size

    # Translate the facial area to the center of the image
    x = (facial_area[0] + facial_area[2]) / 2 - weight / 2
    y = (facial_area[1] + facial_area[3]) / 2 - height / 2

    # Rotate the facial area
    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

    # Translate the facial area back to the original position
    x_new = x_new + weight / 2
    y_new = y_new + height / 2

    # Calculate projected coordinates after alignment
    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2

    # validate projected coordinates are in image's boundaries
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), weight)
    y2 = min(int(y2), height)

    return (x1, y1, x2, y2)


def oracle_extract_face(img: np.ndarray, facial_area, landmarks) -> np.ndarray:
    """The ``align=True`` branch of retina-face 0.0.17 ``extract_faces`` for one face."""
    x = facial_area[0]
    y = facial_area[1]
    w = facial_area[2] - x
    h = facial_area[3] - y

    facial_img = img[y : y + h, x : x + w]

    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    nose = landmarks["nose"]

    # notice that left eye of one is seen on the right from your perspective
    aligned_img, rotate_angle, rotate_direction = alignment_procedure(
        img=img, left_eye=right_eye, right_eye=left_eye, nose=nose
    )

    # find new facial area coordinates after alignment
    rotated_x1, rotated_y1, rotated_x2, rotated_y2 = rotate_facial_area(
        (x, y, x + w, y + h), rotate_angle, rotate_direction, (img.shape[0], img.shape[1])
    )
    facial_img = aligned_img[int(rotated_y1) : int(rotated_y2), int(rotated_x1) : int(rotated_x2)]

    return facial_img[:, :, ::-1]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _image(rng: np.random.Generator, h: int = 120, w: int = 160) -> np.ndarray:
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _landmarks(right_eye, left_eye, nose=(80.0, 70.0), mouth_right=(70.0, 90.0), mouth_left=(90.0, 90.0)):
    """retina-face naming: ``right_eye`` is the eye on the image's left."""
    return {
        "right_eye": list(right_eye),
        "left_eye": list(left_eye),
        "nose": list(nose),
        "mouth_right": list(mouth_right),
        "mouth_left": list(mouth_left),
    }


def _check(img, box, landmarks):
    expected = oracle_extract_face(img, [np.int64(v) for v in box], landmarks)
    got = align_and_crop(img, box, landmarks)
    if expected.size == 0:
        assert got is None
    else:
        assert got is not None
        assert got.dtype == np.uint8
        assert got.shape == expected.shape
        assert np.array_equal(got, expected)
    return expected


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_horizontal_eyes_no_rotation():
    rng = np.random.default_rng(1)
    img = _image(rng)
    box = (40, 30, 120, 110)
    lm = _landmarks(right_eye=(60.0, 55.0), left_eye=(100.0, 55.0))
    crop = _check(img, box, lm)
    # No rotation: the crop is exactly the box, channel-reversed.
    assert np.array_equal(crop, img[30:110, 40:120, ::-1])


def test_eyes_tilted_image_left_eye_higher():
    """Image-left eye (retina ``right_eye``) higher than the image-right one -> direction +1."""
    rng = np.random.default_rng(2)
    img = _image(rng)
    box = (40, 30, 120, 110)
    lm = _landmarks(right_eye=(60.0, 45.0), left_eye=(100.0, 65.0))
    crop = _check(img, box, lm)
    assert crop.size > 0
    # The rotation must actually have changed the pixels.
    assert not np.array_equal(crop, img[30:110, 40:120, ::-1])


def test_eyes_tilted_image_left_eye_lower():
    """Image-left eye lower than the image-right one -> direction -1 (90 - angle branch)."""
    rng = np.random.default_rng(3)
    img = _image(rng)
    box = (40, 30, 120, 110)
    lm = _landmarks(right_eye=(60.0, 65.0), left_eye=(100.0, 45.0))
    crop = _check(img, box, lm)
    assert crop.size > 0
    assert not np.array_equal(crop, img[30:110, 40:120, ::-1])


def test_degenerate_eyes_same_point_c_zero():
    rng = np.random.default_rng(4)
    img = _image(rng)
    box = (40, 30, 120, 110)
    lm = _landmarks(right_eye=(80.0, 55.0), left_eye=(80.0, 55.0))
    crop = _check(img, box, lm)
    assert np.array_equal(crop, img[30:110, 40:120, ::-1])


def test_degenerate_eyes_vertical_b_zero():
    """Same x, image-left eye above: point_3rd == right eye -> b == 0 -> no rotation."""
    rng = np.random.default_rng(5)
    img = _image(rng)
    box = (40, 30, 120, 110)
    lm = _landmarks(right_eye=(80.0, 40.0), left_eye=(80.0, 70.0))
    crop = _check(img, box, lm)
    assert np.array_equal(crop, img[30:110, 40:120, ::-1])


@pytest.mark.parametrize(
    "box",
    [
        (0, 0, 70, 80),  # top-left corner
        (90, 0, 160, 80),  # top-right corner
        (0, 40, 70, 120),  # bottom-left corner
        (90, 40, 160, 120),  # bottom-right corner
        (0, 0, 160, 120),  # whole image
    ],
)
@pytest.mark.parametrize("tilt", [-18.0, 0.0, 12.0])
def test_boxes_touching_borders(box, tilt):
    rng = np.random.default_rng(6)
    img = _image(rng)
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    lm = _landmarks(right_eye=(cx - 15.0, cy - tilt), left_eye=(cx + 15.0, cy + tilt))
    _check(img, box, lm)


def test_float32_landmarks_like_retinaface():
    """retina-face hands over numpy float32 scalars; dtype must flow through unchanged."""
    rng = np.random.default_rng(7)
    img = _image(rng, 200, 180)
    box = (np.int64(50), np.int64(40), np.int64(150), np.int64(170))
    lm = {
        "right_eye": [np.float32(75.3), np.float32(80.1)],
        "left_eye": [np.float32(125.9), np.float32(92.7)],
        "nose": [np.float32(100.0), np.float32(110.0)],
        "mouth_right": [np.float32(80.0), np.float32(140.0)],
        "mouth_left": [np.float32(120.0), np.float32(142.0)],
    }
    _check(img, box, lm)


def test_fuzz_random_boxes_and_landmarks():
    rng = np.random.default_rng(1234)
    for _ in range(60):
        h = int(rng.integers(40, 200))
        w = int(rng.integers(40, 200))
        img = _image(rng, h, w)
        x1 = int(rng.integers(0, w - 5))
        y1 = int(rng.integers(0, h - 5))
        x2 = int(rng.integers(x1 + 1, w + 1))
        y2 = int(rng.integers(y1 + 1, h + 1))
        re = (float(rng.uniform(x1, x2)), float(rng.uniform(y1, y2)))
        le = (float(rng.uniform(x1, x2)), float(rng.uniform(y1, y2)))
        lm = _landmarks(right_eye=re, left_eye=le)
        _check(img, (x1, y1, x2, y2), lm)


def test_empty_box_returns_none():
    rng = np.random.default_rng(8)
    img = _image(rng)
    lm = _landmarks(right_eye=(60.0, 55.0), left_eye=(100.0, 55.0))
    assert align_and_crop(img, (50, 50, 50, 90), lm) is None
    assert align_and_crop(img, (50, 50, 90, 50), lm) is None


def test_output_is_rgb_contiguous_uint8():
    rng = np.random.default_rng(9)
    img = _image(rng)
    lm = _landmarks(right_eye=(60.0, 50.0), left_eye=(100.0, 58.0))
    crop = align_and_crop(img, (40, 30, 120, 110), lm)
    assert crop is not None
    assert crop.flags["C_CONTIGUOUS"]
    assert crop.dtype == np.uint8
    assert crop.ndim == 3 and crop.shape[2] == 3
    # Make sure it is actually channel-reversed relative to the BGR input.
    raw = oracle_extract_face(img, (40, 30, 120, 110), lm)
    assert np.array_equal(crop, raw)


def test_fidelity_matches_retinaface_extract_faces():
    """End-to-end against the real retina-face package (skipped when it is not installed)."""
    pytest.importorskip("retinaface")
    import cv2
    from retinaface import RetinaFace

    from tests.fixtures import require_fixture

    path = require_fixture("obama.jpg")
    img = cv2.imread(str(path))
    assert img is not None

    faces = RetinaFace.detect_faces(img)
    assert isinstance(faces, dict) and len(faces) >= 1

    reference = RetinaFace.extract_faces(img, align=True)
    assert len(reference) == len(faces)

    for (_, face), ref in zip(faces.items(), reference):
        mine = align_and_crop(img, face["facial_area"], face["landmarks"])
        assert mine is not None
        assert mine.shape == ref.shape
        assert np.array_equal(mine, ref)

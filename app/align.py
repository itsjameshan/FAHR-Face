"""Eye-based face alignment: an exact port of retina-face 0.0.17.

``RetinaFace.extract_faces(img, align=True)`` rotates the *whole* image about
its centre so the eyes become horizontal (PIL ``Image.rotate`` with its
defaults: NEAREST resampling, no expansion), rotates the detected box by the
same angle with ``int()`` truncation and clamping, and crops.  The FAHR models
were trained on crops produced this way, so :func:`align_and_crop` reproduces
the geometry bit for bit (``tests/test_align.py`` checks this against the
vendored original).

Only numpy and Pillow are used here so this module can be imported without
torch, OpenCV or TensorFlow.

Landmark naming follows retina-face: ``right_eye`` is the *person's* right eye,
which appears on the image's left.  :func:`alignment_procedure` on the other
hand wants image-left / image-right points, hence the swap in
:func:`align_and_crop`.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

__all__ = [
    "find_euclidean_distance",
    "alignment_procedure",
    "rotate_facial_area",
    "align_and_crop",
]


def find_euclidean_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> float:
    """Euclidean distance between two points (retina-face 0.0.17, verbatim).

    The dtype of the inputs is preserved on purpose: retina-face hands over
    ``np.float32`` landmarks and the angle must be computed the same way.
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def alignment_procedure(
    img: np.ndarray, left_eye: Sequence[float], right_eye: Sequence[float], nose: Optional[Sequence[float]] = None
) -> Tuple[np.ndarray, float, int]:
    """Rotate ``img`` so that the two eyes become horizontal (retina-face 0.0.17).

    Args:
        img: image array (any channel order; it is only rotated).
        left_eye: eye appearing on the *left of the image* (the person's right eye).
        right_eye: eye appearing on the *right of the image* (the person's left eye).
        nose: unused, kept for signature compatibility.

    Returns:
        ``(rotated_img, angle_degrees, direction)`` where ``direction`` is ``-1``
        for a clockwise and ``1`` for a counter-clockwise rotation.  When the eye
        geometry is degenerate the image is returned unchanged with angle ``0.0``.
    """
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # find rotation direction
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # find length of triangle edges
    a = find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = find_euclidean_distance(np.array(right_eye), np.array(left_eye))

    # apply cosine rule
    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)

        # floating point errors can push cos_a slightly outside [-1, 1] -> NaN in arccos
        cos_a = min(1.0, max(-1.0, cos_a))

        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # rotate base image
        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
    else:
        angle = 0.0  # Dummy value for undefined angle

    return img, angle, direction


def rotate_facial_area(
    facial_area: Tuple[int, int, int, int], angle: float, direction: int, size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Rotate the box ``(x1, y1, x2, y2)`` about the image centre (retina-face 0.0.17).

    Args:
        facial_area: box as ``(x1, y1, x2, y2)``.
        angle: rotation angle in degrees.
        direction: ``-1`` clockwise, ``1`` counter-clockwise.
        size: image size as ``(height, width)``.

    Returns:
        The rotated box, truncated with ``int()`` and clamped to the image.
    """
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


def align_and_crop(
    img_bgr: np.ndarray,
    box_xyxy: Sequence[Union[int, float]],
    landmarks: Mapping[str, Sequence[float]],
) -> Optional[np.ndarray]:
    """Align the face in ``img_bgr`` on its eyes and crop it, exactly like retina-face.

    Args:
        img_bgr: ``H x W x 3`` uint8 image in BGR order (OpenCV convention).
        box_xyxy: detected face box ``(x1, y1, x2, y2)``; values are truncated to
            ``int`` as retina-face does with ``astype(int)``.
        landmarks: mapping with at least ``"right_eye"`` and ``"left_eye"`` as
            ``[x, y]`` in retina-face naming (``right_eye`` is on the image's left).
            Values are used with their own dtype so float32 landmarks from
            retina-face give identical angles.

    Returns:
        The aligned crop as an ``h x w x 3`` uint8 **RGB** array, or ``None`` when
        the crop would be empty (zero-area box, or a box rotated out of the image).
    """
    x1, y1, x2, y2 = (int(v) for v in box_xyxy)
    x = x1
    y = y1
    w = x2 - x
    h = y2 - y

    # retina-face names the eyes from the person's point of view; the alignment
    # procedure wants image-left / image-right, hence the swap.
    image_left_eye = landmarks["right_eye"]
    image_right_eye = landmarks["left_eye"]

    aligned_img, rotate_angle, rotate_direction = alignment_procedure(
        img=img_bgr, left_eye=image_left_eye, right_eye=image_right_eye, nose=landmarks.get("nose")
    )

    rotated_x1, rotated_y1, rotated_x2, rotated_y2 = rotate_facial_area(
        (x, y, x + w, y + h), rotate_angle, rotate_direction, (img_bgr.shape[0], img_bgr.shape[1])
    )
    facial_img = aligned_img[int(rotated_y1) : int(rotated_y2), int(rotated_x1) : int(rotated_x2)]

    if facial_img.size == 0:
        return None

    return np.ascontiguousarray(facial_img[:, :, ::-1])

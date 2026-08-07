"""Compute a road-plane homography from four measured point pairs."""

from __future__ import annotations

from traffic_vision.config import Matrix3x3
from traffic_vision.schemas import Point


def _solve_linear_system(matrix: list[list[float]], values: list[float]) -> list[float]:
    """Solve a square linear system with partial-pivot Gaussian elimination."""

    size = len(values)
    augmented = [row[:] + [value] for row, value in zip(matrix, values, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("calibration points do not define a unique homography")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_current
                for current, pivot_current in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [augmented[row][-1] for row in range(size)]


def compute_homography(
    source_points: tuple[Point, Point, Point, Point],
    target_points: tuple[Point, Point, Point, Point],
) -> Matrix3x3:
    """Return the transform from four normalized-image to road-plane points."""

    equations: list[list[float]] = []
    results: list[float] = []
    for source, target in zip(source_points, target_points, strict=True):
        x, y = source.x, source.y
        u, v = target.x, target.y
        equations.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        results.append(u)
        equations.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        results.append(v)

    values = _solve_linear_system(equations, results)
    return (
        (values[0], values[1], values[2]),
        (values[3], values[4], values[5]),
        (values[6], values[7], 1.0),
    )


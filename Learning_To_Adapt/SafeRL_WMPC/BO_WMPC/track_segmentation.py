import numpy as np
from typing import List, Dict, Any


def get_train_segments() -> List[Dict[str, Any]]:
    """
    Get training track segments for optimization.

    Returns:
        List of track segments for testing
    """
    # Define segments for different track types - return simple list for testing
    segments = [
        # Use existing track files only
        {"trajectory": "slider", "start": 0, "end": 500},
        {"trajectory": "test", "start": 0, "end": 300},
    ]

    return segments


def calculate_track_curvature(track_lu_table: np.ndarray) -> np.ndarray:
    """
    Calculate curvature along the track.

    Args:
        track_lu_table: Track lookup table

    Returns:
        np.ndarray: Curvature values
    """
    trackvars = [
        "sval",
        "tval",
        "xtrack",
        "ytrack",
        "phitrack",
        "cos(phi)",
        "sin(phi)",
        "g_upper",
        "g_lower",
    ]

    x = track_lu_table[:, trackvars.index("xtrack")]
    y = track_lu_table[:, trackvars.index("ytrack")]

    # Calculate first and second derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # Calculate curvature: k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx * d2y - dy * d2x)
    denominator = (dx**2 + dy**2) ** (3 / 2)

    # Avoid division by zero
    curvature = np.where(denominator > 1e-10, numerator / denominator, 0)

    return curvature


def segment_track_by_curvature(
    track_lu_table: np.ndarray, curvature_threshold: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Segment track based on curvature.

    Args:
        track_lu_table: Track lookup table
        curvature_threshold: Threshold to distinguish straight from curved

    Returns:
        List of track segments
    """
    curvature = calculate_track_curvature(track_lu_table)

    # Find segments where curvature is above/below threshold
    is_curved = curvature > curvature_threshold

    segments = []
    start_idx = 0
    current_type = is_curved[0]

    for i in range(1, len(is_curved)):
        if is_curved[i] != current_type:
            # End of current segment
            segments.append(
                {
                    "start": start_idx,
                    "end": i,
                    "is_curved": current_type,
                    "avg_curvature": np.mean(curvature[start_idx:i]),
                }
            )
            start_idx = i
            current_type = is_curved[i]

    # Add final segment
    segments.append(
        {
            "start": start_idx,
            "end": len(is_curved),
            "is_curved": current_type,
            "avg_curvature": np.mean(curvature[start_idx:]),
        }
    )

    return segments


def get_segment_groups() -> List[List[Dict[str, Any]]]:
    """
    Get predefined segment groups for optimization.

    Returns:
        List of segment groups
    """
    return get_train_segments()


def validate_segment(segment: Dict[str, Any], track_length: int) -> bool:
    """
    Validate if a segment is within track bounds.

    Args:
        segment: Segment dictionary
        track_length: Total track length

    Returns:
        bool: True if segment is valid
    """
    start = segment.get("start", 0)
    end = segment.get("end", 0)

    # Check bounds
    if start < 0 or end > track_length:
        return False

    # Check segment length
    if end <= start:
        return False

    # Check minimum segment length
    if end - start < 50:  # Minimum 50 points
        return False

    return True


def get_segment_info(
    segment: Dict[str, Any], track_lu_table: np.ndarray
) -> Dict[str, Any]:
    """
    Get additional information about a track segment.

    Args:
        segment: Segment dictionary
        track_lu_table: Track lookup table

    Returns:
        Dict containing segment information
    """
    start = segment["start"]
    end = segment["end"]

    # Extract segment data
    segment_data = track_lu_table[start:end]

    # Calculate segment properties
    trackvars = [
        "sval",
        "tval",
        "xtrack",
        "ytrack",
        "phitrack",
        "cos(phi)",
        "sin(phi)",
        "g_upper",
        "g_lower",
    ]

    x = segment_data[:, trackvars.index("xtrack")]
    y = segment_data[:, trackvars.index("ytrack")]

    # Calculate segment length
    dx = np.diff(x)
    dy = np.diff(y)
    segment_length = np.sum(np.sqrt(dx**2 + dy**2))

    # Calculate average curvature
    curvature = calculate_track_curvature(segment_data)
    avg_curvature = np.mean(curvature)

    # Determine segment type
    segment_type = "curved" if avg_curvature > 0.1 else "straight"

    return {
        "start": start,
        "end": end,
        "length": segment_length,
        "avg_curvature": avg_curvature,
        "type": segment_type,
        "n_points": end - start,
    }

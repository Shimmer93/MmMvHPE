from __future__ import annotations


def select_frame_indices(num_total: int, frame_index: int, num_frames: int) -> list[int]:
    """Select contiguous frame indices from one temporal window."""
    if num_total <= 0:
        return []
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be >= 1, got {num_frames}.")

    if frame_index >= 0:
        start = max(0, min(frame_index, num_total - 1))
    else:
        center = num_total // 2
        start = center - (num_frames // 2)
        start = max(0, min(start, max(0, num_total - num_frames)))
    end = min(start + num_frames, num_total)
    return list(range(start, end))


def select_sample_frame_steps(
    dataset_size: int,
    sample_idx: int,
    temporal_len: int,
    frame_index: int,
    num_frames: int,
) -> list[tuple[int, int]]:
    """Build timeline steps as (sample_index, source_frame_index).

    For temporal windows (temporal_len > 1), steps stay inside one sample.
    For single-frame windows (temporal_len <= 1), steps move across consecutive samples.
    """
    if dataset_size <= 0:
        return []
    sample_idx = max(0, min(sample_idx, dataset_size - 1))

    if temporal_len <= 1:
        if num_frames <= 0:
            raise ValueError(f"`num_frames` must be >= 1, got {num_frames}.")
        start_sample = sample_idx + max(0, frame_index)
        start_sample = max(0, min(start_sample, dataset_size - 1))
        end_sample = min(start_sample + num_frames, dataset_size)
        return [(idx, 0) for idx in range(start_sample, end_sample)]

    return [(sample_idx, fidx) for fidx in select_frame_indices(temporal_len, frame_index, num_frames)]


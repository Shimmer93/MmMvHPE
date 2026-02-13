from __future__ import annotations

import rerun as rr
from pathlib import Path


def init_rerun_session(
    recording_name: str,
    save_rrd: str | None,
    no_serve: bool,
    web_port: int,
    grpc_port: int,
) -> None:
    """Initialize rerun recording, optional file output, and optional live serving."""
    rr.init(recording_name, spawn=False)

    if save_rrd:
        Path(save_rrd).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        rr.save(save_rrd)

    if not no_serve:
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=server_uri)


def init_world_axes() -> None:
    """Log static world/view coordinate systems used by visualization scripts."""
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    rr.log("world/front", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    rr.log("world/side", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)


def set_frame_timeline(frame_idx: int, sample_id: str | None = None) -> None:
    """Set timeline and optional per-frame metadata."""
    rr.set_time("frame", sequence=frame_idx)
    if sample_id:
        rr.log("world/info/sample_id", rr.TextLog(str(sample_id)))

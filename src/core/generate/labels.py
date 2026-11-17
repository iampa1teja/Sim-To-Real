"""Observed-outcome labels and acceptance decisions for generated episodes."""

import numpy as _np

from core.generate.protocols import (FrameLabels as _FrameLabels, Kind as _Kind, RecoveryInfo as _RecoveryInfo,
                                     RejectionReason as _Reason, Segment as _Segment, SegmentLabel as _Label)


def build_imitate(injected):
    return tuple(not value for value in injected)


def frame_labels(stage, injected, ee):
    x, y, z, w = ee.quat_xyzw
    return _FrameLabels(stage, not injected, [*ee.position, w, x, y, z])


def label_columns(labels):
    return {"label.stage": _np.array([labels.stage], dtype=_np.int64),
            "label.imitate": _np.array([int(labels.imitate)], dtype=_np.int64),
            "label.ee_pose": _np.array(labels.ee_pose_wxyz, dtype=_np.float32, copy=True)}


def build_phases(phases):
    runs = []
    for t, name in enumerate(phases):
        if not isinstance(name, str) or not name:
            raise ValueError("Phase names must be non-empty strings")
        if runs and runs[-1]["name"] == name:
            runs[-1]["end"] = t + 1
        else:
            runs.append({"name": name, "start": t, "end": t + 1})
    return runs


def build_recovery(reentries, stage, stages):
    if not reentries:
        return None
    frame, resumed = reentries[-1]
    index = stages.index(resumed)
    back = next((t for t in range(frame, len(stage)) if stage[t] >= index), len(stage))
    return _RecoveryInfo(reentries[0][0], resumed, len(reentries), back)


def build_segments(num_frames, kind, *, onset, injected_end, detected, back_on_plan, phases):
    for boundary in (num_frames, onset, injected_end, detected, back_on_plan):
        if boundary is not None:
            if type(boundary) is not int:
                raise TypeError("Segment boundaries must be integers")
            if boundary < 0:
                raise ValueError("Segment boundaries must be nonnegative")
    if num_frames < 1 or len(phases) != num_frames:
        raise ValueError("Expected one phase per frame and at least one frame")
    kind = _Kind(kind)
    if kind is _Kind.SUCCESS or onset is None:
        rows = [(_Label.NOMINAL, num_frames)]
    elif kind is _Kind.FAILURE:
        rows = [(_Label.NOMINAL, onset), (_Label.PERTURBATION, onset + 1 if injected_end is None else injected_end),
                (_Label.FAILURE, num_frames)]
    else:
        rows = [(_Label.NOMINAL, onset), (_Label.PERTURBATION, num_frames if detected is None else detected),
                (_Label.RECOVERY, num_frames if back_on_plan is None else back_on_plan), (_Label.NOMINAL, num_frames)]
    segments, start = [], 0
    for label, boundary in rows:
        end = max(start, min(boundary, num_frames))
        if end > start:
            segments.append(_Segment(label, start, end, phases[start]))
        start = end
    return tuple(segments)


def accept(kind, *, success, applied, reentered, reentered_before_onset):
    kind = _Kind(kind)
    if kind is _Kind.SUCCESS:
        if not success:
            return _Reason.EXPERT_FAILED
        return _Reason.UNPLANNED_REGRESSION if reentered else None
    if not applied:
        return _Reason.PERTURBATION_NOT_APPLIED
    if reentered_before_onset:
        return _Reason.UNPLANNED_REGRESSION
    if kind is _Kind.FAILURE:
        return _Reason.PERTURBATION_INEFFECTIVE if success else None
    if not success:
        return _Reason.RECOVERY_FAILED
    return None if reentered else _Reason.NO_REGRESSION

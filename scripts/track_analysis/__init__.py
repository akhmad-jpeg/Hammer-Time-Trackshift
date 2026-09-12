"""Trackshift Track Analysis — single-lap overtake visualization and detection.

Modules:
    fastf1_loader         – FastF1 session/lap/telemetry loading
    telemetry_alignment   – distance-based driver alignment + interpolation
    overtake_detector     – spatial separation, closing phase, overtake detection
    track_map             – circuit visualization with overtake annotations
    models                – dataclasses for structured results
"""
